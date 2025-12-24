package telemetry

import (
	"context"
	"fmt"
	"os"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.37.0"
)

// Config holds the configuration for OpenTelemetry
type Config struct {
	ServiceName   string
	OTLPEndpoint  string
	OTLPInsecure  bool
	SamplingRatio float64
}

// InitTracer initializes OpenTelemetry tracer provider
func InitTracer(ctx context.Context, cfg Config) (*sdktrace.TracerProvider, error) {
	// Create resource with service information
	res, err := resource.Merge(
		resource.Default(),
		resource.NewWithAttributes(
			semconv.SchemaURL,
			semconv.ServiceName(cfg.ServiceName),
		),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create resource: %w", err)
	}

	// Create OTLP exporter
	var exporter *otlptrace.Exporter
	if cfg.OTLPEndpoint != "" {
		opts := []otlptracegrpc.Option{
			otlptracegrpc.WithEndpoint(cfg.OTLPEndpoint),
			otlptracegrpc.WithTimeout(5 * time.Second),
		}

		if cfg.OTLPInsecure {
			opts = append(opts, otlptracegrpc.WithInsecure())
		}

		client := otlptracegrpc.NewClient(opts...)
		exporter, err = otlptrace.New(ctx, client)
		if err != nil {
			return nil, fmt.Errorf("failed to create OTLP exporter: %w", err)
		}
	}

	// Create tracer provider with the exporter
	var opts []sdktrace.TracerProviderOption
	opts = append(opts, sdktrace.WithResource(res))

	if exporter != nil {
		opts = append(opts, sdktrace.WithBatcher(exporter))
	}

	// Add sampler based on environment and config
	samplingRatio := cfg.SamplingRatio

	if samplingRatio >= 1.0 {
		opts = append(opts, sdktrace.WithSampler(sdktrace.AlwaysSample()))
	} else if samplingRatio <= 0 {
		opts = append(opts, sdktrace.WithSampler(sdktrace.NeverSample()))
	} else {
		opts = append(opts, sdktrace.WithSampler(sdktrace.ParentBased(sdktrace.TraceIDRatioBased(samplingRatio))))
	}

	tp := sdktrace.NewTracerProvider(opts...)

	// Set global tracer provider
	otel.SetTracerProvider(tp)

	// Set global propagator for trace context
	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(
		propagation.TraceContext{},
		propagation.Baggage{},
	))

	return tp, nil
}

// ConfigFromEnv creates a Config from environment variables
func ConfigFromEnv(serviceName string) Config {
	return Config{
		ServiceName:   serviceName,
		OTLPEndpoint:  getEnvOrDefault("OTEL_EXPORTER_OTLP_ENDPOINT", "localhost:4317"),
		OTLPInsecure:  getEnvOrDefault("OTEL_EXPORTER_OTLP_INSECURE", "true") == "true",
		SamplingRatio: 1.0,
	}
}

func getEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}
