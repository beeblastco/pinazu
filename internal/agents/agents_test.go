package agents

import (
	"fmt"
	"os"

	"github.com/joho/godotenv"
	"github.com/pinazu/internal/service"
)

func newMockServiceConfig() *service.ExternalDependenciesConfig {
	if err := godotenv.Load("../../.env"); err != nil {
		fmt.Printf("Error loading .env file: %s\n", err)
		fmt.Println("Using environment variables from the system")
	}
	return &service.ExternalDependenciesConfig{
		Debug: true,
		Http:  nil,
		Nats: &service.NatsConfig{
			URL:                    os.Getenv("NATS_URL"),
			JetStreamDefaultConfig: nil,
		},
		Database: &service.DatabaseConfig{
			Host:     os.Getenv("POSTGRES_HOST"),
			Port:     os.Getenv("POSTGRES_PORT"),
			User:     os.Getenv("POSTGRES_USER"),
			Password: os.Getenv("POSTGRES_PASSWORD"),
			Dbname:   os.Getenv("POSTGRES_DB"),
			SSLMode:  "disable",
		},
		Tracing: nil,
		LLMConfig: &service.LLMConfig{
			Bedrock: &service.BedrockLLMServiceConfig{
				CredentialType:  "default",
				Region:          "ap-southeast-1",
				AccessKeyID:     os.Getenv("AWS_ACCESS_KEY_ID"),
				SecretAccessKey: os.Getenv("AWS_SECRET_ACCESS_KEY"),
				SessionToken:    os.Getenv("AWS_SESSION_TOKEN"),
			},
			Google: &service.GoogleLLMServiceConfig{
				APIKey: os.Getenv("GOOGLE_API_KEY"),
			},
		},
	}
}

var MockServiceConfigs = newMockServiceConfig()
