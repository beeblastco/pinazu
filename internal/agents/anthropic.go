package agents

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/packages/param"
	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/pinazu/internal/db"
	"github.com/pinazu/internal/service"
)

// handleAnthropicRequest handles requests for Anthropic models
func (as *AgentService) handleAnthropicRequest(m []anthropic.MessageParam, spec *AgentSpecs, header *service.EventHeaders, meta *service.EventMetadata) (*anthropic.MessageParam, string, error) {
	// Initialize variables to accumulate content
	var (
		signature, toolUseID, toolName                                          string
		stop                                                                    anthropic.StopReason
		response                                                                anthropic.MessageParam
		accumulatedThinkContent, accumulatedTextContent, accumulatedToolContent strings.Builder
		content                                                                 []anthropic.ContentBlockParamUnion
		tools                                                                   []anthropic.ToolUnionParam
		subAgentList                                                            []db.Agent
	)

	// Fetch sub agent for this agent
	if spec.SubAgents != nil && len(spec.SubAgents.Allows) > 0 {
		queries := db.New(as.s.GetDB())
		for _, subAgentID := range spec.SubAgents.Allows {
			// Parse string to UUID
			subAgentUUID, err := uuid.Parse(subAgentID)
			if err != nil {
				as.log.Warn("Invalid UUID format for sub-agent ID", "sub_agent_id", subAgentID, "error", err)
				continue
			}

			// Check if agent ID is valid
			agent, err := queries.GetAgentByID(as.ctx, subAgentUUID)
			if err != nil {
				if err == pgx.ErrNoRows {
					as.log.Error("Sub-agent ID not found in database, skipping this agent", "sub_agent_id", subAgentID)
					continue
				}
				as.log.Error("Failed to check sub-agent ID validity", "sub_agent_id", subAgentID, "error", err)
				continue
			}

			// Update the valid sub agent list
			subAgentList = append(subAgentList, agent)

			as.log.Debug("Valid sub-agent found", "sub_agent_id", subAgentID)
		}

		// Get the invoke_agent tool
		invokeAgentToolID, _ := uuid.Parse("550e8400-c00b-8888-3333-446655447896")
		invokeAgentTool, err := queries.GetToolById(as.ctx, invokeAgentToolID)
		if err != nil {
			as.log.Error("Failed to get invoke_agent tool", "error", err)
			return nil, "", fmt.Errorf("failed to get invoke_agent tool: %w", err)
		}

		// Extract description
		description := ""
		if invokeAgentTool.Description.Valid {
			description = invokeAgentTool.Description.String
		}

		// Convert tool config to parameter schema following existing pattern
		var inputSchema map[string]any
		switch invokeAgentTool.Config.Type {
		case db.ToolTypeInternal:
			internalConfig := invokeAgentTool.Config.GetInternal()
			if internalConfig != nil {
				// Convert OpenAPI schema to map for Anthropic
				schemaBytes, err := json.Marshal(internalConfig.Params)
				if err != nil {
					as.log.Error("Failed to marshal invoke_agent tool schema", "tool_name", invokeAgentTool.Name, "error", err)
					return nil, "", fmt.Errorf("failed to marshal invoke_agent tool schema: %w", err)
				}
				if err := json.Unmarshal(schemaBytes, &inputSchema); err != nil {
					as.log.Error("Failed to unmarshal invoke_agent tool schema", "tool_name", invokeAgentTool.Name, "error", err)
					return nil, "", fmt.Errorf("failed to unmarshal invoke_agent tool schema: %w", err)
				}
			}
		default:
			as.log.Error("invoke_agent tool is not of internal type", "actual_type", invokeAgentTool.Config.Type)
			return nil, "", fmt.Errorf("invoke_agent tool is not of internal type")
		}

		// Create enum values for valid sub-agent IDs
		validSubAgentStrings := make([]string, len(subAgentList))
		for i, agent := range subAgentList {
			validSubAgentStrings[i] = agent.ID.String()
		}

		// Update the agent_id parameter enum in the existing schema
		if properties, ok := inputSchema["properties"].(map[string]any); ok {
			if agentIDProp, exists := properties["agent_id"].(map[string]any); exists {
				agentIDProp["enum"] = validSubAgentStrings
			}
		}

		// Extract properties and required fields from the updated schema
		var properties any
		var required []string

		if props, exists := inputSchema["properties"]; exists {
			properties = props
		}
		if req, exists := inputSchema["required"]; exists {
			if reqSlice, ok := req.([]any); ok {
				required = make([]string, len(reqSlice))
				for i, r := range reqSlice {
					if reqStr, ok := r.(string); ok {
						required[i] = reqStr
					}
				}
			}
		}

		// Create the tool using the database schema with enum constraint
		t := &anthropic.ToolParam{
			Name:        invokeAgentTool.Name,
			Description: param.NewOpt(description),
			InputSchema: anthropic.ToolInputSchemaParam{
				Type:       "object",
				Properties: properties,
				Required:   required,
			},
		}
		tools = append(tools, anthropic.ToolUnionParam{
			OfTool: t,
		})
	}

	// Fetch and convert tools for this agent
	if len(spec.ToolRefs) > 0 {
		var err error
		tools, err = as.fetchAnthropicTools(spec.ToolRefs)
		if err != nil {
			as.log.Error("Failed to convert tools to Anthropic format", "error", err)
			return nil, "", fmt.Errorf("failed to convert tools to Anthropic format: %w", err)
		}

		as.log.Debug("Loaded tools for agent", "tool_count", len(tools), "tool_names", func() []string {
			names := make([]string, len(tools))
			for i, t := range tools {
				names[i] = t.OfTool.Name
			}
			return names
		}())
	}

	// Create the request parameters for the Anthropic API
	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(spec.Model.ModelID),
		MaxTokens: spec.Model.MaxTokens,
		Messages:  m,
		System:    getSystemPrompt(spec, subAgentList),
	}

	// Include tools if available
	if len(tools) > 0 {
		params.Tools = tools
	}

	// Conditionally include Thinking or Structure Ouput
	if len(spec.Model.ResponseFormat) > 0 && !spec.Model.Thinking.Enabled {
		// Add the prefill assistant response to start JSON output
		prefillMsg := anthropic.NewAssistantMessage(anthropic.NewTextBlock("{"))
		params.Messages = append(params.Messages, prefillMsg)
	} else if len(spec.Model.ResponseFormat) > 0 && spec.Model.Thinking.Enabled {
		// Modify the system prompt to response with JSON output
		params.System = getSystemForThinkingAndStructureOutput(spec)
		// Add the prefill assistant response to start JSON output
		prefillMsg := anthropic.NewAssistantMessage(anthropic.NewTextBlock("{"))
		params.Messages = append(params.Messages, prefillMsg)
	} else if len(spec.Model.ResponseFormat) == 0 && spec.Model.Thinking.Enabled {
		// Conditionally include Thinking if enabled
		params.Thinking = *getThinkingConfig(spec)
	}

	// Conditionally include Temperature if provided by user
	if spec.Model.Temperature != 0 {
		params.Temperature = param.NewOpt(spec.Model.Temperature)
	}

	// Conditionally include TopP if provided by user
	if spec.Model.TopP != 0 {
		params.TopP = param.NewOpt(spec.Model.TopP)
	}

	// Conditionally include TopK if provided by user
	if spec.Model.TopK != 0 {
		params.TopK = param.NewOpt(spec.Model.TopK)
	}

	if spec.ToolChoice != (ToolChoice{}) {
		switch spec.ToolChoice.Type {
		case "none":
			params.ToolChoice = anthropic.ToolChoiceUnionParam{
				OfNone: &anthropic.ToolChoiceNoneParam{
					Type: "none",
				},
			}
		case "auto":
			params.ToolChoice = anthropic.ToolChoiceUnionParam{
				OfAuto: &anthropic.ToolChoiceAutoParam{
					Type:                   "auto",
					DisableParallelToolUse: param.NewOpt(spec.ToolChoice.DisableParallelToolUse),
				},
			}
		case "any":
			params.ToolChoice = anthropic.ToolChoiceUnionParam{
				OfAny: &anthropic.ToolChoiceAnyParam{
					Type:                   "any",
					DisableParallelToolUse: param.NewOpt(spec.ToolChoice.DisableParallelToolUse),
				},
			}
		case "tool":
			params.ToolChoice = anthropic.ToolChoiceUnionParam{
				OfTool: &anthropic.ToolChoiceToolParam{
					Type:                   "tool",
					Name:                   spec.ToolChoice.Name,
					DisableParallelToolUse: param.NewOpt(spec.ToolChoice.DisableParallelToolUse),
				},
			}
		}
	}

	paramBytes, _ := json.Marshal(params)
	as.log.Debug("Show invoke params", "params", string(paramBytes))

	if spec.Model.Stream {
		stream := as.ac.Messages.NewStreaming(as.ctx, params)

		as.log.Debug("Streaming response from Anthropic API")
		for stream.Next() {
			event := stream.Current()

			// Publish the streaming event to websocket client
			as.publishAnthropicStreamEvent(event, header, meta)

			// Continue processing the stream
			switch event.Type {
			case "message_start":
			case "content_block_start":
				switch event.ContentBlock.Type {
				case "thinking":
				case "redacted_thinking":
				case "text":
				case "tool_use":
					toolUseID = event.ContentBlock.ID
					as.log.Debug("Recieve tool use block with", "id", toolUseID)
					toolName = event.ContentBlock.Name
				case "signature":
				case "server_tool_use":
				case "web_search_tool_result":
				default:
					as.log.Warn("Unknown content block start type", "type", event.ContentBlock.Type)
				}
			case "content_block_delta":
				switch event.Delta.Type {
				case "thinking_delta":
					accumulatedThinkContent.WriteString(event.Delta.Thinking)
				case "signature_delta":
					signature = event.Delta.Signature
				case "text_delta":
					accumulatedTextContent.WriteString(event.Delta.Text)
				case "input_json_delta":
					accumulatedToolContent.WriteString(event.Delta.PartialJSON)
					as.log.Debug("Received content block delta json", "delta", event.Delta.PartialJSON)
				case "server_tool_use":
				default:
					as.log.Warn("Unknown content block delta type", "type", event.Delta.Type)
				}
			case "content_block_stop":
				// Add completed content blocks to the response
				if accumulatedThinkContent.Len() > 0 {
					thinkBlock := anthropic.NewThinkingBlock(signature, accumulatedThinkContent.String())
					// Set the Type field for streaming responses
					if thinkBlock.OfThinking != nil {
						thinkBlock.OfThinking.Type = "thinking"
					}
					content = append(content, thinkBlock)
					accumulatedThinkContent.Reset()
				}
				if accumulatedTextContent.Len() > 0 {
					textBlock := anthropic.NewTextBlock(accumulatedTextContent.String())
					// Set the Type field for streaming responses
					if textBlock.OfText != nil {
						textBlock.OfText.Type = "text"
					}
					content = append(content, textBlock)
					accumulatedTextContent.Reset()
				}
				if accumulatedToolContent.Len() > 0 {
					var jsonData map[string]any
					// Validate the JSON
					err := json.Unmarshal([]byte(accumulatedToolContent.String()), &jsonData)
					if err != nil {
						return nil, "", fmt.Errorf("invalid JSON in tool use: %w", err)
					}
					content = append(content, anthropic.NewToolUseBlock(toolUseID, jsonData, toolName))
					// Print last content for debugging
					as.log.Debug("Completed tool use content", "content", content[len(content)-1])
					accumulatedToolContent.Reset()
				}
			case "message_delta":
				stop = event.Delta.StopReason
			case "message_stop":
				// Extract Amazon Bedrock invocation metrics from raw JSON
				var rawEvent map[string]any
				if err := json.Unmarshal([]byte(event.RawJSON()), &rawEvent); err != nil {
					return nil, "", fmt.Errorf("failed to parse raw response JSON: %w", err)
				} else if bedrockMetrics, ok := rawEvent["amazon-bedrock-invocationMetrics"]; ok {
					if metrics, ok := bedrockMetrics.(map[string]any); ok {
						as.log.Info("Amazon Bedrock invocation metrics",
							"input_token_count", metrics["inputTokenCount"],
							"output_token_count", metrics["outputTokenCount"],
							"invocation_latency", metrics["invocationLatency"],
							"first_byte_latency", metrics["firstByteLatency"],
						)
					}
				}
			default:
				as.log.Warn("Unknown event type received", "event_type", event.Type)
			}
		}

		if err := stream.Err(); err != nil && err != io.EOF {
			return nil, "", fmt.Errorf("streaming error: %w", err)
		}

	} else {
		resp, err := as.ac.Messages.New(as.ctx, params)
		if err != nil {
			return nil, "", fmt.Errorf("failed to create message: %w", err)
		}
		// Extract Amazon Bedrock invocation metrics from raw JSON for non-streaming
		var rawResp map[string]any
		if err := json.Unmarshal([]byte(resp.RawJSON()), &rawResp); err != nil {
			return nil, "", fmt.Errorf("failed to parse raw response JSON: %w", err)
		} else if bedrockMetrics, ok := rawResp["amazon-bedrock-invocationMetrics"]; ok {
			if metrics, ok := bedrockMetrics.(map[string]any); ok {
				as.log.Info("Amazon Bedrock invocation metrics",
					"input_token_count", metrics["inputTokenCount"],
					"output_token_count", metrics["outputTokenCount"],
					"invocation_latency", metrics["invocationLatency"],
					"first_byte_latency", metrics["firstByteLatency"],
				)
			}
		}

		content = resp.ToParam().Content
		stop = resp.StopReason
	}

	// Create response message with accumulated content
	response = anthropic.MessageParam{
		Role:    "assistant",
		Content: content,
	}

	return &response, string(stop), nil
}

// getSystemPrompt returns the system prompt for the agent based on the provided specs
func getSystemPrompt(spec *AgentSpecs, subAgentList []db.Agent) []anthropic.TextBlockParam {
	systemText := spec.System

	// Add schema instruction if response format is specified
	if len(spec.Model.ResponseFormat) > 0 {
		schemaBytes, err := json.Marshal(spec.Model.ResponseFormat)
		if err == nil {
			schemaInstruction := fmt.Sprintf("\n\nYou must respond with valid JSON that matches this exact schema:\n%s\n\n", string(schemaBytes))
			systemText += schemaInstruction
		}
	}

	// Add sub-agent list if there are sub-agents
	if len(subAgentList) > 0 {
		var builder strings.Builder
		builder.WriteString("\n\n You have access to the following sub-agents: \n")
		// Build system prompt string for each agent description
		for _, agent := range subAgentList {
			builder.WriteString(" - ")
			builder.WriteString(agent.ID.String())
			builder.WriteString(":\n   Name: ")
			builder.WriteString(agent.Name)
			builder.WriteString(":\n   Description: ")
			builder.WriteString(agent.Description.String)
		}
		systemText += builder.String()
	}

	textBlock := anthropic.TextBlockParam{
		Type: "text",
		Text: systemText,
		CacheControl: anthropic.CacheControlEphemeralParam{
			Type: "ephemeral",
		},
	}

	return []anthropic.TextBlockParam{textBlock}
}

// getSystemForThinkingAndStructureOutput returns the system prompt for the agent with thinking and structure output
func getSystemForThinkingAndStructureOutput(spec *AgentSpecs) []anthropic.TextBlockParam {
	systemText := spec.System

	newResponseFormat := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"thought": map[string]string{
				"type":        "string",
				"description": "A thought to think about, it should be a step by step thinking for the problem.",
			},
			"answer": spec.Model.ResponseFormat,
		},
		"required": []string{"thinking", "answer"},
	}

	schemaBytes, err := json.Marshal(newResponseFormat)
	if err == nil {
		schemaInstruction := fmt.Sprintf("\n\nYou must respond with valid JSON that matches this exact schema:\n%s\n\n", string(schemaBytes))
		systemText += schemaInstruction
	}

	textBlock := anthropic.TextBlockParam{
		Type: "text",
		Text: systemText,
		CacheControl: anthropic.CacheControlEphemeralParam{
			Type: "ephemeral",
		},
	}

	return []anthropic.TextBlockParam{textBlock}
}

// getThinkingConfig returns the thinking configuration for the agent based on the provided specs
func getThinkingConfig(spec *AgentSpecs) *anthropic.ThinkingConfigParamUnion {
	var thinkingConfig anthropic.ThinkingConfigParamUnion
	if spec.Model.Thinking.Enabled {
		thinkingConfig = anthropic.ThinkingConfigParamUnion{
			OfEnabled: &anthropic.ThinkingConfigEnabledParam{
				BudgetTokens: spec.Model.Thinking.BudgetToken,
				Type:         "enabled",
			},
		}
	} else {
		thinkingConfig = anthropic.ThinkingConfigParamUnion{
			OfDisabled: &anthropic.ThinkingConfigDisabledParam{
				Type: "disabled",
			},
		}
	}
	return &thinkingConfig
}

// publishAnthropicStreamEvent publishes Anthropic stream events to WebSocket clients
func (as *AgentService) publishAnthropicStreamEvent(event anthropic.MessageStreamEventUnion, header *service.EventHeaders, meta *service.EventMetadata) {
	// Convert anthropic event to WebsocketResponseEventMessage
	wsEvent := ToWebsocketResponseEventMessage(event, db.ProviderModelAnthropic)

	// Publish to NATS for WebSocket handler
	newEvent := service.NewEvent(wsEvent, header, &service.EventMetadata{
		TraceID:   meta.TraceID,
		Timestamp: time.Now().UTC(),
	})

	// Publish using the new PublishWithUser method for WebSocket events
	if err := newEvent.PublishWithUser(as.s.GetNATS(), header.UserID); err != nil {
		as.log.Error("Failed to publish websocket event", "error", err)
		return
	}
}

// fetchAgentTools retrieves tools from database based on agent's tool_refs
func (as *AgentService) fetchAnthropicTools(toolRefs []uuid.UUID) ([]anthropic.ToolUnionParam, error) {
	if len(toolRefs) == 0 {
		return nil, nil
	}

	// Fetch tools from database
	queries := db.New(as.s.GetDB())
	tools, err := queries.GetToolsByIDs(as.ctx, toolRefs)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch tools from database: %w", err)
	}

	// Log warnings for any missing tools
	as.logMissingTools(tools, toolRefs)

	// Convert database tools to Anthropic tool parameters
	return as.convertToolsToAnthropicParams(tools)
}

// logMissingTools checks if any requested tools were not found and logs warnings
func (as *AgentService) logMissingTools(tools []db.Tool, toolRefs []uuid.UUID) {
	if len(tools) >= len(toolRefs) {
		return
	}

	foundToolIDs := make(map[uuid.UUID]bool)
	for _, tool := range tools {
		foundToolIDs[tool.ID] = true
	}

	for _, toolRef := range toolRefs {
		if !foundToolIDs[toolRef] {
			as.log.Warn("Tool not found in database, will not use this tool", "tool_id", toolRef)
		}
	}
}

// convertToolsToAnthropicParams converts database tools to Anthropic tool parameters
func (as *AgentService) convertToolsToAnthropicParams(tools []db.Tool) ([]anthropic.ToolUnionParam, error) {
	var anthropicTools []anthropic.ToolUnionParam

	for _, tool := range tools {
		anthropicTool, err := as.convertSingleToolToAnthropicParam(tool)
		if err != nil {
			// Log the error but continue processing other tools
			as.log.Warn("Failed to convert tool", "tool_name", tool.Name, "error", err)
			continue
		}
		if anthropicTool != nil {
			anthropicTools = append(anthropicTools, *anthropicTool)
		}
	}

	return anthropicTools, nil
}

// convertSingleToolToAnthropicParam converts a single database tool to Anthropic tool parameter
func (as *AgentService) convertSingleToolToAnthropicParam(tool db.Tool) (*anthropic.ToolUnionParam, error) {
	// Extract description
	description := ""
	if tool.Description.Valid {
		description = tool.Description.String
	}

	// Convert tool config to parameter schema
	inputSchema, err := as.extractToolInputSchema(tool)
	if err != nil {
		return nil, err
	}

	if inputSchema == nil {
		return nil, nil
	}

	// Create Anthropic tool parameter
	toolParam := as.createAnthropicToolParam(tool.Name, description, inputSchema)
	return &anthropic.ToolUnionParam{OfTool: toolParam}, nil
}

// extractToolInputSchema extracts input schema from tool config based on tool type
func (as *AgentService) extractToolInputSchema(tool db.Tool) (map[string]any, error) {
	switch tool.Config.Type {
	case db.ToolTypeStandalone:
		return as.marshalToolParams(tool.Config.GetStandalone().Params, string(db.ToolTypeStandalone))
	case db.ToolTypeWorkflow:
		return as.marshalToolParams(tool.Config.GetWorkflow().Params, string(db.ToolTypeWorkflow))
	case db.ToolTypeInternal:
		return as.marshalToolParams(tool.Config.GetInternal().Params, string(db.ToolTypeInternal))
	case db.ToolTypeMCP:
		as.log.Debug("Skipping MCP tool - dynamic schema discovery required", "tool_name", tool.Name)
		return nil, nil
	default:
		as.log.Warn("Unknown tool type", "tool_name", tool.Name, "type", tool.Config.Type)
		return nil, nil
	}
}

// marshalToolParams marshals and unmarshals tool parameters to convert to map[string]any
func (as *AgentService) marshalToolParams(params any, toolType string) (map[string]any, error) {
	if params == nil {
		return nil, nil
	}

	schemaBytes, err := json.Marshal(params)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal %s tool schema: %w", toolType, err)
	}

	var inputSchema map[string]any
	if err := json.Unmarshal(schemaBytes, &inputSchema); err != nil {
		return nil, fmt.Errorf("failed to unmarshal %s tool schema: %w", toolType, err)
	}

	return inputSchema, nil
}

// createAnthropicToolParam creates an Anthropic tool parameter from schema
func (as *AgentService) createAnthropicToolParam(name, description string, inputSchema map[string]any) *anthropic.ToolParam {
	// Extract properties and required fields from the schema
	var properties any
	var required []string

	if props, exists := inputSchema["properties"]; exists {
		properties = props
	}
	if req, exists := inputSchema["required"]; exists {
		if reqSlice, ok := req.([]any); ok {
			required = make([]string, len(reqSlice))
			for i, r := range reqSlice {
				if reqStr, ok := r.(string); ok {
					required[i] = reqStr
				}
			}
		}
	}

	return &anthropic.ToolParam{
		Name:        name,
		Description: param.NewOpt(description),
		InputSchema: anthropic.ToolInputSchemaParam{
			Type:       "object",
			Properties: properties,
			Required:   required,
		},
		CacheControl: anthropic.CacheControlEphemeralParam{
			Type: "ephemeral",
		},
	}
}
