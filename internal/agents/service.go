package agents

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/credentials/stscreds"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/sts"
	"github.com/google/uuid"
	"github.com/hashicorp/go-hclog"
	"github.com/nats-io/nats.go"
	"github.com/openai/openai-go"
	"github.com/pinazu/internal/db"
	"github.com/pinazu/internal/service"
	"google.golang.org/genai"
	"gopkg.in/yaml.v3"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/bedrock"
)

type (
	AgentService struct {
		ac  *anthropic.Client
		gc  *genai.Client
		oc  *openai.Client
		bc  *bedrockruntime.Client
		s   service.Service
		log hclog.Logger
		wg  *sync.WaitGroup
		ctx context.Context
		// State tracking for Bedrock streaming event normalization
		contentBlockStartSent map[int64]bool
	}

	AgentSpecs struct {
		Model      ModelSpecs  `yaml:"model"`
		System     string      `yaml:"system"`
		ToolRefs   []uuid.UUID `yaml:"tool_refs,omitempty"`
		ToolChoice ToolChoice  `yaml:"tool_choice,omitempty"`
		SubAgents  *SubAgents  `yaml:"sub_agents,omitempty"`
	}

	SubAgents struct {
		Configs SubAgentConfigs `yaml:"configs,omitempty"`
		Allows  []string        `yaml:"allows,omitempty"`
	}

	SubAgentConfigs struct {
		SharedMemory bool `yaml:"shared_memory,omitempty"`
	}

	ModelSpecs struct {
		Provider       string         `yaml:"provider"`
		ModelID        string         `yaml:"model_id"`
		MaxTokens      int64          `yaml:"max_tokens"`
		Temperature    float64        `yaml:"temperature"`
		TopP           float64        `yaml:"top_p"`
		TopK           int64          `yaml:"top_k"`
		Thinking       ThinkingSpecs  `yaml:"thinking"`
		Stream         bool           `yaml:"stream"`
		ResponseFormat map[string]any `yaml:"response_format"`
	}

	ThinkingSpecs struct {
		Enabled     bool  `yaml:"enabled"`
		BudgetToken int64 `yaml:"budget_token"`
	}

	ToolChoice struct {
		Type                   string `yaml:"type"`
		Name                   string `yaml:"name,omitempty"`
		DisableParallelToolUse bool   `yaml:"disable_parallel_tool_use,omitempty"`
	}
)

func NewService(ctx context.Context, externalDependenciesConfig *service.ExternalDependenciesConfig, log hclog.Logger, wg *sync.WaitGroup) (*AgentService, error) {
	if externalDependenciesConfig == nil {
		return nil, fmt.Errorf("externalDependenciesConfig is nil")
	}
	cfg, err := config.LoadDefaultConfig(ctx)
	if err != nil {
		log.Warn("failed to load AWS configuration, %v", err)
	}
	if externalDependenciesConfig.LLMConfig != nil && externalDependenciesConfig.LLMConfig.Bedrock != nil {
		// Prepare options for loading AWS config
		optFns := []func(*config.LoadOptions) error{
			config.WithRegion(externalDependenciesConfig.LLMConfig.Bedrock.Region),
		}

		switch externalDependenciesConfig.LLMConfig.Bedrock.CredentialType {
		case "assume_role":
			log.Info("Using Assume Role Credential Type for Bedrock LLM Service")
			// Create a properly configured AWS config for STS client with region
			stsConfig, err := config.LoadDefaultConfig(ctx, config.WithRegion(externalDependenciesConfig.LLMConfig.Bedrock.Region))
			if err != nil {
				log.Warn("failed to load AWS configuration for STS client: %v", err)
			}
			optFns = append(optFns, config.WithCredentialsProvider(stscreds.NewAssumeRoleProvider(
				sts.NewFromConfig(stsConfig),
				externalDependenciesConfig.LLMConfig.Bedrock.AssumeRole,
				func(o *stscreds.AssumeRoleOptions) {
					o.Duration = time.Minute * 15
					o.RoleSessionName = "pinazu-bedrock-" + uuid.New().String()
				},
			)))
		case "default":
			log.Info("Using Default Credential Type from environment variables for Bedrock LLM Service")
			// Load credentials from environment variables (using standard AWS environment variable names)
			if externalDependenciesConfig.LLMConfig.Bedrock.AccessKeyID != "" && externalDependenciesConfig.LLMConfig.Bedrock.SecretAccessKey != "" {
				log.Debug("Loading static credentials from environment variables")
				optFns = append(optFns, config.WithCredentialsProvider(
					credentials.NewStaticCredentialsProvider(
						externalDependenciesConfig.LLMConfig.Bedrock.AccessKeyID,
						externalDependenciesConfig.LLMConfig.Bedrock.SecretAccessKey,
						externalDependenciesConfig.LLMConfig.Bedrock.SessionToken,
					),
				))
			} else {
				log.Warn("AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY environment variables not set, falling back to default credential chain")
			}
		}

		cfg, err = config.LoadDefaultConfig(ctx, optFns...)
		if err != nil {
			log.Warn("failed to load AWS configuration %v", err)
		}
	}

	// Create a new Anthropic client
	ac := anthropic.NewClient(bedrock.WithConfig(cfg))

	// Create a new Google AI client
	var gc *genai.Client
	if externalDependenciesConfig.LLMConfig != nil && externalDependenciesConfig.LLMConfig.Google != nil {
		gc, err = genai.NewClient(ctx, &genai.ClientConfig{APIKey: externalDependenciesConfig.LLMConfig.Google.APIKey})
		if err != nil {
			log.Warn("failed to create Google AI client: %v", err)
		}
	} else {
		// Fallback to environment variable for backward compatibility
		gc, err = genai.NewClient(ctx, &genai.ClientConfig{APIKey: os.Getenv("GOOGLE_AI_API_KEY")})
		if err != nil {
			log.Warn("failed to create Google AI client: %v", err)
		}
	}

	// Create a new Bedrock client
	bc := bedrockruntime.NewFromConfig(cfg)

	// Create a new OpenAI client
	oc := openai.NewClient()

	// Create a new service instance
	config := &service.Config{
		Name:                 "agents-handler-service",
		Version:              "0.0.1",
		Description:          "Agent service for handling agent execution.",
		ExternalDependencies: externalDependenciesConfig,
		ErrorHandler:         nil,
	}
	s, err := service.NewService(ctx, config)
	if err != nil {
		return nil, fmt.Errorf("failed to create agent service: %v", err)
	}

	as := &AgentService{ac: &ac, gc: gc, oc: &oc, bc: bc, s: s, log: log, wg: wg, ctx: ctx}

	s.RegisterHandler(service.AgentInvokeEventSubject.String(), as.invokeEventCallback)
	s.RegisterHandler("v1.svc.agent._info", nil)
	s.RegisterHandler("v1.svc.agent._stats", nil)

	// Start a goroutine to wait for context cancellation and then shutdown
	go func() {
		<-ctx.Done()
		as.log.Warn("Agent service shutting down...")
		if err := as.s.Shutdown(); err != nil {
			as.log.Error("Error during agent service shutdown", "error", err)
		}
		as.wg.Done()
	}()

	return as, nil
}

// invokeEventCallback handles the agent invoke request event callback
func (as *AgentService) invokeEventCallback(msg *nats.Msg) {
	// Check if context was cancelled
	select {
	case <-as.ctx.Done():
		as.log.Info("Context cancelled, stopping message processing")
		return
	default:
	}

	// Start a new span for the callback
	_, span := as.s.GetTracer().Start(as.ctx, "invokeCallback")
	defer span.End()

	// Parse NATS message to request struct
	req, err := service.ParseEvent[*service.AgentInvokeEventMessage](msg.Data)
	if err != nil {
		as.log.Error("Failed to unmarshal message to request", "error", err)
		return
	}

	// Handle the callback logic here
	as.log.Info("Received and validated agent invoke message",
		"agent_id", req.Msg.AgentId,
		"thread_id", req.H.ThreadID,
		"connection_id", req.H.ConnectionID,
		"user_id", req.H.UserID,
	)

	// Load the agent specs
	queries := db.New(as.s.GetDB())
	yamlSpecs, err := queries.GetAgentSpecsByID(as.ctx, req.Msg.AgentId)
	if err != nil {
		if err.Error() == "no rows in result set" {
			as.log.Error("Agent not found", "agent_id", req.Msg.AgentId)
			err := fmt.Errorf("invalid agent_id")
			service.NewErrorEvent[*service.WebsocketResponseEventMessage](req.H, req.M, err).PublishWithUser(as.s.GetNATS(), req.H.UserID)
			service.NewErrorEvent[*service.TaskFinishEventMessage](req.H, req.M, err).Publish(as.s.GetNATS())
		} else {
			as.log.Error("Failed to load agent specs", "error", err)
			err := fmt.Errorf("failed to load agent specs: %w", err)
			service.NewErrorEvent[*service.WebsocketResponseEventMessage](req.H, req.M, err).PublishWithUser(as.s.GetNATS(), req.H.UserID)
			service.NewErrorEvent[*service.TaskFinishEventMessage](req.H, req.M, err).Publish(as.s.GetNATS())
		}
		return
	}

	// Convert specs to AgentSpecs struct
	specs := &AgentSpecs{}
	err = yaml.Unmarshal([]byte(yamlSpecs.String), specs)
	if err != nil {
		as.log.Error("Failed to unmarshal agent specs", "error", err)
		return
	}

	// Detect the model provider from the model string
	as.log.Debug("Detected model provider", "provider", specs.Model.Provider, "model", specs.Model.ModelID)

	// Route to appropriate handler based on provider using generics
	var response any
	var stop string
	switch specs.Model.Provider {
	case "bedrock/anthropic":
		// Parse Anthropic messages
		msgs, err := ParseMessages[anthropic.MessageParam](req.Msg.Messages)
		if err != nil {
			// Log error and create error message
			as.log.Error("Failed to parse Anthropic messages", "error", err)
			err = fmt.Errorf("failed to parse Anthropic messages: %w", err)

			// Create and publish new Error Event back to websocket
			service.NewErrorEvent[*service.WebsocketResponseEventMessage](req.H, req.M, err).PublishWithUser(as.s.GetNATS(), req.H.UserID)
			service.NewErrorEvent[*service.TaskFinishEventMessage](req.H, req.M, err).Publish(as.s.GetNATS())
			return
		}

		// Invoke the Anthropic model
		response, stop, err = as.handleAnthropicRequest(msgs, specs, req.H, req.M)
		if err != nil {
			// Log error and create error message
			as.log.Error("Failed to handle Anthropic request", "error", err)
			err = fmt.Errorf("failed to handle Anthropic request: %w", err)

			// Create and publish new Error Event back to websocket
			service.NewErrorEvent[*service.WebsocketResponseEventMessage](req.H, req.M, err).PublishWithUser(as.s.GetNATS(), req.H.UserID)
			service.NewErrorEvent[*service.TaskFinishEventMessage](req.H, req.M, err).Publish(as.s.GetNATS())
			return
		}

	case "bedrock":
		// Parse Anthropic messages (consistent format)
		msgs, err := ParseMessages[anthropic.MessageParam](req.Msg.Messages)
		if err != nil {
			// Log error and create error message
			as.log.Error("Failed to parse Anthropic messages", "error", err)
			err = fmt.Errorf("failed to parse Anthropic messages: %w", err)

			// Create and publish new Error Event back to websocket
			service.NewErrorEvent[*service.WebsocketResponseEventMessage](req.H, req.M, err).PublishWithUser(as.s.GetNATS(), req.H.UserID)
			service.NewErrorEvent[*service.TaskFinishEventMessage](req.H, req.M, err).Publish(as.s.GetNATS())
			return
		}

		// Invoke the Bedrock Foundation model
		response, stop, err = as.handleBedrockRequest(msgs, specs, req.H, req.M)
		if err != nil {
			// Log error and create error message
			as.log.Error("Failed to handle Bedrock request", "error", err)
			err = fmt.Errorf("failed to handle Bedrock request: %w", err)

			// Create and publish new Error Event back to websocket
			service.NewErrorEvent[*service.WebsocketResponseEventMessage](req.H, req.M, err).PublishWithUser(as.s.GetNATS(), req.H.UserID)
			service.NewErrorEvent[*service.TaskFinishEventMessage](req.H, req.M, err).Publish(as.s.GetNATS())
			return
		}

	case "openai":
		// Parse OpenAI messages
		msgs, err := ParseMessages[openai.ChatCompletionMessageParamUnion](req.Msg.Messages)
		if err != nil {
			// Log error and create error message
			as.log.Error("Failed to parse OpenAI messages", "error", err)
			err = fmt.Errorf("failed to parse OpenAI messages: %w", err)

			// Create and publish new Error Event back to websocket
			service.NewErrorEvent[*service.WebsocketResponseEventMessage](req.H, req.M, err).PublishWithUser(as.s.GetNATS(), req.H.UserID)
			service.NewErrorEvent[*service.TaskFinishEventMessage](req.H, req.M, err).Publish(as.s.GetNATS())
			return
		}

		// Invoke the OpenAI model
		response, err = as.handleOpenAIRequest(msgs, specs, req.H)
		if err != nil {
			// Log error and create error message
			as.log.Error("Failed to handle OpenAI request", "error", err)
			err = fmt.Errorf("failed to handle OpenAI request: %w", err)

			// Create and publish new Error Event back to websocket
			service.NewErrorEvent[*service.WebsocketResponseEventMessage](req.H, req.M, err).PublishWithUser(as.s.GetNATS(), req.H.UserID)
			service.NewErrorEvent[*service.TaskFinishEventMessage](req.H, req.M, err).Publish(as.s.GetNATS())
			return
		}

	case "google":
		// Parse Anthropic messages (consistent format)
		msgs, err := ParseMessages[anthropic.MessageParam](req.Msg.Messages)
		if err != nil {
			// Log error and create error message
			as.log.Error("Failed to parse Anthropic messages", "error", err)
			err = fmt.Errorf("failed to parse Anthropic messages: %w", err)

			// Create and publish new Error Event back to websocket
			service.NewErrorEvent[*service.WebsocketResponseEventMessage](req.H, req.M, err).PublishWithUser(as.s.GetNATS(), req.H.UserID)
			service.NewErrorEvent[*service.TaskFinishEventMessage](req.H, req.M, err).Publish(as.s.GetNATS())
			return
		}

		// Invoke the Gemini model
		response, stop, err = as.handleGeminiRequest(msgs, specs, req.H, req.M)
		if err != nil {
			// Log error and create error message
			as.log.Error("Failed to handle Gemini request", "error", err)
			err = fmt.Errorf("failed to handle Gemini request: %w", err)

			// Create and publish new Error Event back to websocket
			service.NewErrorEvent[*service.WebsocketResponseEventMessage](req.H, req.M, err).PublishWithUser(as.s.GetNATS(), req.H.UserID)
			service.NewErrorEvent[*service.TaskFinishEventMessage](req.H, req.M, err).Publish(as.s.GetNATS())
			return
		}

	default:
		as.log.Error("Unsupported model provider", "provider", specs.Model.Provider)
		return
	}

	// Convert response to db.JsonRaw
	responseBytes, err := json.Marshal(response)
	if err != nil {
		as.log.Error("Failed to marshal response", "error", err)
		return
	}

	switch stop {
	case "end_turn":
		event := service.NewEvent(&service.TaskFinishEventMessage{
			AgentId:     req.Msg.AgentId,
			RecipientId: req.Msg.RecipientId,
			Response:    responseBytes,
		}, req.H, &service.EventMetadata{
			TraceID:   req.M.TraceID,
			Timestamp: time.Now().UTC(),
		})
		err = event.Publish(as.s.GetNATS())
		if err != nil {
			as.log.Error("Failed to publish event", "error", err)
			service.NewErrorEvent[*service.TaskFinishEventMessage](req.H, req.M, err).Publish(as.s.GetNATS())
			return
		}
	case "tool_use":
		event := service.NewEvent(&service.ToolDispatchEventMessage{
			AgentId:     req.Msg.AgentId,
			Provider:    db.ProviderModel(specs.Model.Provider),
			RecipientId: req.Msg.RecipientId,
			Message:     responseBytes,
		}, req.H, &service.EventMetadata{
			TraceID:   req.M.TraceID,
			Timestamp: time.Now().UTC(),
		})
		err = event.Publish(as.s.GetNATS())
		if err != nil {
			as.log.Error("Failed to publish event", "error", err)
			service.NewErrorEvent[*service.ToolDispatchEventMessage](req.H, req.M, err).Publish(as.s.GetNATS())
			return
		}
	default:
		// Handle unexpected stop reasons
		as.log.Warn("Unexpected stop reason", "stop_reason", stop)
		service.NewErrorEvent[*service.TaskFinishEventMessage](req.H, req.M, fmt.Errorf("unexpected stop reason: %s", stop)).Publish(as.s.GetNATS())
	}
}
