package service

import (
	"fmt"
	"os"
	"strings"

	"github.com/hashicorp/go-hclog"
	"github.com/pinazu/internal/logger"
	"github.com/pinazu/internal/telemetry"
	"github.com/urfave/cli/v3"
	"gopkg.in/yaml.v3"
)

type (
	// Config is the configuration of a service.
	Config struct {
		// Name represents the name of the service.
		Name string `json:"name"`

		// Version is a SemVer compatible version string.
		Version string `json:"version"`

		// Description of the service.
		Description string `json:"description"`

		// ExternalDependencies contains the configuration for external dependencies client
		ExternalDependencies *ExternalDependenciesConfig `json:"external_dependencies"`

		// ErrorHandler is invoked on any nats-related service error.
		ErrorHandler ErrHandler
	}

	// ExternalDependenciesConfig represents the configuration for external dependencies.
	ExternalDependenciesConfig struct {
		Debug     bool              `yaml:"debug"`
		Http      *HttpServerConfig `yaml:"http"`
		Nats      *NatsConfig       `yaml:"nats"`
		Database  *DatabaseConfig   `yaml:"database"`
		Tracing   *TracingConfig    `yaml:"tracing"`
		Storage   *StorageConfig    `yaml:"storage"`
		Cache     *CacheConfig      `yaml:"cache"`
		LLMConfig *LLMConfig        `yaml:"llm_config"`
	}

	// CacheType represents the type of caching system to use
	CacheType string

	HttpServerConfig struct {
		Port string `yaml:"port"`
	}

	// NatsConfig represents the configuration for NATS server.
	NatsConfig struct {
		URL                    string           `yaml:"url"`
		JetStreamDefaultConfig *JetStreamConfig `yaml:"jetstream_default_config"`
	}

	// JetStreamConfig represents the configuration for JetStream streams.
	JetStreamConfig struct {
		MaxMsgs       int64 `yaml:"max_msgs"`        // Maximum number of messages per stream
		MaxBytes      int64 `yaml:"max_bytes"`       // Maximum bytes per stream
		MaxAgeSeconds int   `yaml:"max_age_seconds"` // Maximum age in seconds
		Replicas      int   `yaml:"replicas"`        // Number of replicas
		MaxDeliver    int   `yaml:"max_deliver"`     // Maximum delivery attempts for consumers
	}

	// DatabaseConfig represents the configuration for the database.
	// Support only postgres for now.
	DatabaseConfig struct {
		Host     string `yaml:"host"`
		Port     string `yaml:"port"`
		User     string `yaml:"user"`
		Password string `yaml:"password"`
		Dbname   string `yaml:"dbname"`
		SSLMode  string `yaml:"sslmode"` // e.g., "disable", "require", "verify-ca", "verify-full". Certain DB setup may require SSL mode, e.g. AWS RDS 17+ need "require".
	}

	// TracingConfig represents the configuration for OpenTelemetry tracing.
	TracingConfig struct {
		ServiceName      string  `yaml:"service_name"`
		ExporterEndpoint string  `yaml:"exporter_endpoint"`
		ExporterInsecure bool    `yaml:"exporter_insecure"`
		SamplingRatio    float64 `yaml:"sampling_ratio"`
	}

	// StorageConfig represents the configuration for storage backends.
	StorageConfig struct {
		S3 *S3Config `yaml:"s3"`
	}

	// CacheConfig represents the configuration for caching.
	CacheConfig struct {
		Type   CacheType `yaml:"type"`   // CacheTypeMemory or CacheTypeS3
		Bucket string    `yaml:"bucket"` // S3 bucket for caching (when type is CacheTypeS3)
	}

	// S3Config represents the configuration for S3-compatible storage.
	S3Config struct {
		EndpointURL       string `yaml:"endpoint_url"`        // Custom endpoint for MinIO/S3-compatible services
		AccessKeyID       string `yaml:"access_key_id"`       // Access key ID (for static credentials)
		SecretAccessKey   string `yaml:"secret_access_key"`   // Secret access key (for static credentials)
		Region            string `yaml:"region"`              // AWS region
		UsePathStyle      bool   `yaml:"use_path_style"`      // Use path-style URLs (true for MinIO, false for AWS S3)
		AssumeRoleARN     string `yaml:"assume_role_arn"`     // ARN of the role to assume (for assume role auth)
		AssumeRoleSession string `yaml:"assume_role_session"` // Session name for assume role (optional)
		CredentialType    string `yaml:"credential_type"`     // "static" or "assume_role" or "default"
	}

	LLMConfig struct {
		Bedrock *BedrockLLMServiceConfig `yaml:"bedrock"`
		Google  *GoogleLLMServiceConfig  `yaml:"google"`
	}

	// A separation for configuration in order to overcome the Quota limit put by AWS on various Bedrock services.
	// Most of the time should not be required, but in the case of new accounts, a quota of 2 requests/min is not servicable for any type of LLM use cases.
	BedrockLLMServiceConfig struct {
		CredentialType  string `yaml:"type"`              // default -> will use the default credential chain, assume role will use the AssumeRole credential chain
		AssumeRole      string `yaml:"assume_role"`       // Role to assume when making calls to AWS for bedrock service
		AccessKeyID     string `yaml:"access_key_id"`     // Access key ID (for static credentials)
		SecretAccessKey string `yaml:"secret_access_key"` // Secret access key (for static credentials)
		SessionToken    string `yaml:"session_token"`
		Region          string `yaml:"region"` // AWS region for the bedrock service. It may differ from default region for other system, namely S3.
	}

	// GoogleLLMServiceConfig represents the configuration for Google AI services.
	GoogleLLMServiceConfig struct {
		APIKey string `yaml:"api_key"` // API key for Google AI services
	}
)

const (
	// CacheTypeMemory uses in-memory caching (cleared when flow completes)
	CacheTypeMemory CacheType = "memory"

	// CacheTypeS3 uses S3/MinIO for persistent caching
	CacheTypeS3 CacheType = "s3"
)

// String returns the string representation of CacheType
func (ct CacheType) String() string {
	return string(ct)
}

// IsValid checks if the CacheType is valid
func (ct CacheType) IsValid() bool {
	switch ct {
	case CacheTypeMemory, CacheTypeS3:
		return true
	default:
		return false
	}
}

// UnmarshalYAML implements custom YAML unmarshaling with validation
func (ct *CacheType) UnmarshalYAML(value *yaml.Node) error {
	var str string
	if err := value.Decode(&str); err != nil {
		return err
	}

	cacheType := CacheType(strings.ToLower(str))
	if !cacheType.IsValid() {
		return fmt.Errorf("invalid cache type '%s'. Valid options are: %s, %s",
			str, CacheTypeMemory, CacheTypeS3)
	}

	*ct = cacheType
	return nil
}

// LoadExternalConfigFile loads the ExternalDependencies configuration from an external file.
// It returns the configuration and an error if any.
// If provided cmd, the ExternalDependencies configuration will override the configuration with the flags provided in the command line if they are set.
func LoadExternalConfigFile(cfgPath string, cmd *cli.Command) (*ExternalDependenciesConfig, error) {
	cfg := &ExternalDependenciesConfig{}

	if cfgPath != "" {
		if _, err := os.Stat(cfgPath); err == nil {
			data, err := os.ReadFile(cfgPath)
			if err != nil {
				return nil, err
			}

			// Expand environment variables in the YAML content
			expandedData := os.ExpandEnv(string(data))

			err = yaml.Unmarshal([]byte(expandedData), cfg)
			if err != nil {
				return nil, err
			}

			// Validate cache configuration
			if err := cfg.ValidateCacheConfig(); err != nil {
				return nil, fmt.Errorf("cache configuration validation failed: %w", err)
			}
		}
	}

	// If no command is provided, return the config as is
	if cmd == nil {
		return cfg, nil
	}
	cfg = cfg.mergeFlags(cmd)
	return cfg, nil
}

// MergeFlags merges the ExternalDependenciesConfig with the flags provided in the command line.
// If a flag is set, it overrides the value in the config file.
func (ec *ExternalDependenciesConfig) mergeFlags(cmd *cli.Command) *ExternalDependenciesConfig {
	port := getCommandString(cmd, "port")
	natsURL := getCommandString(cmd, "nats-url")
	dbHost := getCommandString(cmd, "db-host")
	dbPort := getCommandString(cmd, "db-port")
	dbUser := getCommandString(cmd, "db-user")
	dbPass := getCommandString(cmd, "db-password")
	dbName := getCommandString(cmd, "db-name")

	if port != "" {
		if ec.Http == nil {
			ec.Http = &HttpServerConfig{}
		}
		ec.Http.Port = port
	}
	if natsURL != "" {
		if ec.Nats == nil {
			ec.Nats = &NatsConfig{}
		}
		ec.Nats.URL = natsURL
	}
	if dbHost != "" {
		if ec.Database == nil {
			ec.Database = &DatabaseConfig{}
		}
		ec.Database.Host = dbHost
	}
	if dbPort != "" {
		if ec.Database == nil {
			ec.Database = &DatabaseConfig{}
		}
		ec.Database.Port = dbPort
	}
	if dbUser != "" {
		if ec.Database == nil {
			ec.Database = &DatabaseConfig{}
		}
		ec.Database.User = dbUser
	}
	if dbPass != "" {
		if ec.Database == nil {
			ec.Database = &DatabaseConfig{}
		}
		ec.Database.Password = dbPass
	}
	if dbName != "" {
		if ec.Database == nil {
			ec.Database = &DatabaseConfig{}
		}
		ec.Database.Dbname = dbName
	}

	// Set default SSL mode if database config exists but no SSL mode is set
	if ec.Database != nil && ec.Database.SSLMode == "" {
		ec.Database.SSLMode = "disable"
	}

	return ec
}

// Valid checks if the configuration is valid.
func (c *Config) valid() error {
	if !nameRegexp.MatchString(c.Name) {
		return fmt.Errorf("invalid service name: %s", c.Name)
	}
	if !semVerRegexp.MatchString(c.Version) {
		return fmt.Errorf("invalid service version: %s", c.Version)
	}
	return nil
}

// getDatabaseConnectionString returns the database connection string based on the configuration.
func (c *Config) getDatabaseConnectionString() string {
	if c.ExternalDependencies.Database == nil {
		return "host=localhost port=5432 user=postgres password= dbname=postgres sslmode=disable"
	}
	return fmt.Sprintf("host=%s port=%s user=%s password=%s dbname=%s sslmode=%s",
		c.ExternalDependencies.Database.Host,
		c.ExternalDependencies.Database.Port,
		c.ExternalDependencies.Database.User,
		c.ExternalDependencies.Database.Password,
		c.ExternalDependencies.Database.Dbname,
		c.ExternalDependencies.Database.SSLMode,
	)
}

// getOpenTelemetryConfig returns the OpenTelemetry configuration based on the configuration.
func (c *Config) getOpenTelemetryConfig() *telemetry.Config {
	if c.ExternalDependencies.Tracing == nil {
		return &telemetry.Config{
			ServiceName:   c.Name,
			OTLPEndpoint:  "localhost:4317",
			OTLPInsecure:  true,
			SamplingRatio: 1.0,
		}
	}
	return &telemetry.Config{
		ServiceName:   fmt.Sprintf("%s-%s", c.ExternalDependencies.Tracing.ServiceName, c.Name),
		OTLPEndpoint:  c.ExternalDependencies.Tracing.ExporterEndpoint,
		OTLPInsecure:  c.ExternalDependencies.Tracing.ExporterInsecure,
		SamplingRatio: c.ExternalDependencies.Tracing.SamplingRatio,
	}
}

// GetJetStreamConfig returns the JetStream configuration.
// If no config is provided, it returns nil and the caller should handle loading defaults from config file.
func (nc *NatsConfig) GetJetStreamConfig() *JetStreamConfig {
	if nc == nil {
		return nil
	}
	return nc.JetStreamDefaultConfig
}

// GetLogLevel returns the appropriate log level based on the debug setting.
// If debug is true, returns Debug level, otherwise returns Info level.
func (ec *ExternalDependenciesConfig) GetLogLevel() hclog.Level {
	if ec.Debug {
		return hclog.Debug
	}
	return hclog.Info
}

// CreateLogger creates a new logger with the appropriate log level based on the debug configuration.
func (ec *ExternalDependenciesConfig) CreateLogger() hclog.Logger {
	return logger.NewCustomColorLogWithLevel(ec.GetLogLevel())
}

// ValidateCacheConfig validates the cache configuration
func (ec *ExternalDependenciesConfig) ValidateCacheConfig() error {
	if ec.Cache == nil {
		// No cache config is fine - will use default behavior
		return nil
	}

	cacheConfig := ec.Cache

	// Validate that cache type is set
	if cacheConfig.Type == "" {
		return fmt.Errorf("cache type must be specified. Valid options are: %s, %s",
			CacheTypeMemory, CacheTypeS3)
	}

	// Validate S3 cache requirements
	if cacheConfig.Type == CacheTypeS3 {
		if cacheConfig.Bucket == "" {
			return fmt.Errorf("cache bucket must be specified when using S3 cache type")
		}

		if ec.Storage == nil || ec.Storage.S3 == nil {
			return fmt.Errorf("S3 storage configuration is required when using S3 cache type")
		}
	}

	return nil
}

// getCommandString helper function to get the string value from the command line.
func getCommandString(cmd any, name string) string {
	type stringGetter interface {
		String(string) string
	}
	if sg, ok := cmd.(stringGetter); ok {
		return sg.String(name)
	}
	return ""
}
