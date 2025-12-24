package service

import (
	"context"
	"errors"
	"fmt"
	"regexp"
	"sync"
	"sync/atomic"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/nats-io/nats.go"
	"github.com/nats-io/nuid"
	"github.com/pinazu/internal/telemetry"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/trace"

	sdktrace "go.opentelemetry.io/otel/sdk/trace"
)

type (
	// Service exposes methods to operate on a service instance.
	Service interface {
		// RegisterHandler registers a NATS message handler for a specific subject.
		RegisterHandler(string, nats.MsgHandler)

		// Stop drains the endpoint and all monitoring endpoints,
		// unsubscribes from all subscriptions and marks the service as stopped.
		Shutdown() error

		// Info returns the service info.
		Info() Info

		// Stats returns statistics for the service endpoint and all monitoring endpoints.
		Stats() Stats

		// GetDB returns the database connection pool.
		GetDB() *pgxpool.Pool

		// GetNATS returns the NATS connection.
		GetNATS() *nats.Conn

		// GetTracer returns the tracer for the service.
		GetTracer() trace.Tracer
	}

	// ErrHandler is a function used to configure a custom error handler for a service,
	ErrHandler func(Service, *NATSError)

	// NATSError represents an error returned by a NATS Subscription.
	NATSError struct {
		Subject     string
		Description string
		err         error
	}

	// ServiceIdentity contains fields helping to identity a service instance.
	ServiceIdentity struct {
		Name    string `json:"name"`
		ID      string `json:"id"`
		Version string `json:"version"`
	}

	// Stats is the type returned by service stats monitoring.
	Stats struct {
		ServiceIdentity
		Type          string                   `json:"type"`
		Started       time.Time                `json:"started"`
		Subscriptions []*SubscriptionStatsInfo `json:"subscriptions"`
	}

	// SubscriptionStatsBase contains common fields for subscription stats.
	SubscriptionStatsBase struct {
		Subject   string `json:"subject"`
		LastError string `json:"last_error"`
	}

	// SubscriptionStatsInfo contains stats snapshot for JSON serialization.
	SubscriptionStatsInfo struct {
		SubscriptionStatsBase
		NumMessages uint64 `json:"num_messages"`
		NumErrors   uint64 `json:"num_errors"`
	}

	// SubscriptionStats contains internal stats with atomic counters.
	SubscriptionStats struct {
		SubscriptionStatsBase
		NumMessages atomic.Uint64
		NumErrors   atomic.Uint64
	}

	// Info is the basic information about a service type.
	Info struct {
		ServiceIdentity
		Type          string             `json:"type"`
		Description   string             `json:"description"`
		Subscriptions []SubscriptionInfo `json:"subscriptions"`
	}

	// SubscriptionInfo contains info for a subscription.
	SubscriptionInfo struct {
		Subject string `json:"subject"`
	}

	service struct {
		// Config contains service configuration
		Config

		// External dependencies
		nc            *nats.Conn
		pool          *pgxpool.Pool
		traceProvider *sdktrace.TracerProvider
		tracer        trace.Tracer

		// Internal state
		id       string
		ctx      context.Context
		cancel   context.CancelFunc
		mu       sync.RWMutex
		workerWg sync.WaitGroup
		started  time.Time
		stopped  bool

		// Subscriptions and handlers
		subscriptions []*nats.Subscription
		handlers      map[string]nats.MsgHandler
		stats         map[string]*SubscriptionStats
	}
)

// Regular expressions for validation
var (
	semVerRegexp = regexp.MustCompile(`^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$`)
	nameRegexp   = regexp.MustCompile(`^[A-Za-z0-9\-_]+$`)
)

// Response types
const (
	InfoResponseType  = "io.nats.micro.v1.info_response"
	StatsResponseType = "io.nats.micro.v1.stats_response"
)

func NewService(ctx context.Context, config *Config, natsOptions ...nats.Option) (Service, error) {
	if err := config.valid(); err != nil {
		return nil, err
	}

	// Connect to NATS server
	natsURL := "nats://localhost:4222" // default
	if config.ExternalDependencies.Nats != nil && config.ExternalDependencies.Nats.URL != "" {
		natsURL = config.ExternalDependencies.Nats.URL
	}

	// Add connection options for better reliability
	natsOptions = append(natsOptions,
		nats.Timeout(10*time.Second),      // Increase connection timeout from default 2s
		nats.RetryOnFailedConnect(true),   // Enable retry on failed connect
		nats.MaxReconnects(5),             // Try up to 5 times
		nats.ReconnectWait(2*time.Second), // Wait 2s between retries
		nats.DisconnectErrHandler(func(nc *nats.Conn, err error) {
			if err != nil {
				fmt.Printf("NATS disconnected: %v\n", err)
			}
		}),
		nats.ReconnectHandler(func(nc *nats.Conn) {
			fmt.Printf("NATS reconnected to %s\n", nc.ConnectedUrl())
		}),
	)

	nc, err := nats.Connect(natsURL, natsOptions...)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to NATS server: %w", err)
	}

	// Connect to the database
	pool, err := pgxpool.New(ctx, config.getDatabaseConnectionString())
	if err != nil {
		return nil, fmt.Errorf("failed to connect to database: %w", err)
	}

	// Create new OpenTelemetry tracer
	traceProvider, err := telemetry.InitTracer(ctx, *config.getOpenTelemetryConfig())
	if err != nil {
		return nil, fmt.Errorf("failed to initialize tracer: %w", err)
	}
	tracer := otel.Tracer(config.Name)

	// Create context with cancel
	serviceCtx, cancel := context.WithCancel(ctx)

	// Generate unique service ID
	id := nuid.Next()

	// Create new service instance
	svc := &service{
		Config:        *config,
		nc:            nc,
		pool:          pool,
		traceProvider: traceProvider,
		tracer:        tracer,
		id:            id,
		ctx:           serviceCtx,
		cancel:        cancel,
		started:       time.Now().UTC(),
		subscriptions: make([]*nats.Subscription, 0),
		handlers:      make(map[string]nats.MsgHandler),
		stats:         make(map[string]*SubscriptionStats),
	}

	return svc, nil
}

// RegisterHandler registers a NATS message handler for a specific subject.
func (s *service) RegisterHandler(subject string, handler nats.MsgHandler) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.stopped {
		return
	}

	s.handlers[subject] = handler

	// Initialize stats for this subject
	s.stats[subject] = &SubscriptionStats{
		// Target the embedded struct
		SubscriptionStatsBase: SubscriptionStatsBase{
			Subject: subject,
		},
	}

	// Create subscription with error handling wrapper
	sub, err := s.nc.Subscribe(subject, func(msg *nats.Msg) {
		s.workerWg.Add(1)
		go func() {
			defer s.workerWg.Done()

			// Check if context was cancelled
			select {
			case <-s.ctx.Done():
				return
			default:
			}

			// Update stats through atomic operations
			if stat, exists := s.stats[subject]; exists {
				stat.NumMessages.Add(1)
			}

			// Handle the message
			handler(msg)
		}()
	})

	if err != nil {
		if s.ErrorHandler != nil {
			s.ErrorHandler(s, &NATSError{
				Subject:     subject,
				Description: err.Error(),
				err:         err,
			})
		}
		return
	}

	s.subscriptions = append(s.subscriptions, sub)
}

// Shutdown drains all subscriptions, stops all workers and marks the service as stopped.
func (s *service) Shutdown() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.stopped {
		return nil
	}

	// Cancel context to stop all workers
	s.cancel()

	// Drain all subscriptions
	for _, sub := range s.subscriptions {
		if err := sub.Drain(); err != nil {
			// If connection is closed, draining is not possible but we continue cleanup
			if !errors.Is(err, nats.ErrConnectionClosed) && s.ErrorHandler != nil {
				s.ErrorHandler(s, &NATSError{
					Subject:     sub.Subject,
					Description: fmt.Sprintf("failed to drain subscription: %v", err),
					err:         err,
				})
			}
		}
	}

	// Clear subscriptions
	s.subscriptions = nil
	s.handlers = make(map[string]nats.MsgHandler)

	// Wait for all workers to finish
	s.workerWg.Wait()

	// Close NATS connection
	if s.nc != nil {
		s.nc.Drain()
	}

	// Close database connection
	if s.pool != nil {
		s.pool.Close()
	}

	// Shutdown tracer provider
	if s.traceProvider != nil {
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		s.traceProvider.Shutdown(shutdownCtx)
	}

	s.stopped = true
	return nil
}

// Info returns information about the service
func (s *service) Info() Info {
	s.mu.RLock()
	defer s.mu.RUnlock()

	subscriptions := make([]SubscriptionInfo, 0, len(s.handlers))
	for subject := range s.handlers {
		subscriptions = append(subscriptions, SubscriptionInfo{
			Subject: subject,
		})
	}

	return Info{
		ServiceIdentity: ServiceIdentity{
			Name:    s.Config.Name,
			ID:      s.id,
			Version: s.Config.Version,
		},
		Type:          InfoResponseType,
		Description:   s.Config.Description,
		Subscriptions: subscriptions,
	}
}

// Stats returns statistics for the service endpoint and all monitoring endpoints.
func (s *service) Stats() Stats {
	s.mu.RLock()
	defer s.mu.RUnlock()

	subscriptions := make([]*SubscriptionStatsInfo, 0, len(s.stats))
	for _, stat := range s.stats {
		// Create a clean stats snapshot with loaded atomic values
		statsInfo := &SubscriptionStatsInfo{
			SubscriptionStatsBase: stat.SubscriptionStatsBase,
			NumMessages:           stat.NumMessages.Load(),
			NumErrors:             stat.NumErrors.Load(),
		}
		subscriptions = append(subscriptions, statsInfo)
	}

	return Stats{
		ServiceIdentity: ServiceIdentity{
			Name:    s.Config.Name,
			ID:      s.id,
			Version: s.Config.Version,
		},
		Type:          StatsResponseType,
		Started:       s.started,
		Subscriptions: subscriptions,
	}
}

// Stopped returns whether the service has been stopped.
func (s *service) Stopped() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.stopped
}

// GetDB returns the database connection pool.
func (s *service) GetDB() *pgxpool.Pool {
	return s.pool
}

// GetNATS returns the NATS connection.
func (s *service) GetNATS() *nats.Conn {
	return s.nc
}

// GetTracer returns the tracer for the service.
func (s *service) GetTracer() trace.Tracer {
	return s.tracer
}

func (e *NATSError) Error() string {
	return fmt.Sprintf("%q: %s", e.Subject, e.Description)
}

func (e *NATSError) Unwrap() error {
	return e.err
}
