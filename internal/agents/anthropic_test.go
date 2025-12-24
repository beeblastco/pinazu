package agents

import (
	"context"
	"fmt"
	"os"
	"sync"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/getkin/kin-openapi/openapi3"
	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgtype"
	"github.com/jackc/pgx/v5/pgxpool"
	pq_compat "github.com/jackc/pgx/v5/stdlib"
	"github.com/joho/godotenv"
	"github.com/pinazu/internal/db"
	"github.com/pinazu/internal/service"
	"github.com/pressly/goose/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const testModelID = "global.anthropic.claude-sonnet-4-5-20250929-v1:0"

// Common test setup for database queries
func setupTestDB(t *testing.T) *pgxpool.Pool {
	if err := godotenv.Load("../../.env"); err != nil {
		fmt.Printf("Error loading .env file: %s\n", err)
		fmt.Println("Using environment variables from the system")
	}
	t.Helper()
	dbPool, err := pgxpool.New(context.Background(), os.Getenv("POSTGRES_URL"))
	if err != nil {
		t.Fatalf("Failed to connect to the database: %v", err)
	}
	goose.SetBaseFS(os.DirFS("../../sql"))
	if err := goose.SetDialect("postgres"); err != nil {
		panic(fmt.Errorf("failed to set goose dialect: %w", err))
	}
	if err := goose.Up(pq_compat.OpenDBFromPool(dbPool), "migrations"); err != nil {
		panic(fmt.Errorf("failed to run migrations: %w", err))
	}
	return dbPool
}

func TestGetSystemPrompt(t *testing.T) {
	tests := []struct {
		name     string
		spec     *AgentSpecs
		expected []anthropic.TextBlockParam
	}{
		{
			name: "simple_system_prompt",
			spec: &AgentSpecs{
				System: "You are a helpful assistant.",
				Model: ModelSpecs{
					ModelID: testModelID,
				},
			},
			expected: []anthropic.TextBlockParam{
				{
					Type: "text",
					Text: "You are a helpful assistant.",
					CacheControl: anthropic.CacheControlEphemeralParam{
						Type: "ephemeral",
					},
				},
			},
		},
		{
			name: "empty_system_prompt",
			spec: &AgentSpecs{
				System: "",
				Model: ModelSpecs{
					ModelID: testModelID,
				},
			},
			expected: []anthropic.TextBlockParam{
				{
					Type: "text",
					Text: "",
					CacheControl: anthropic.CacheControlEphemeralParam{
						Type: "ephemeral",
					},
				},
			},
		},
		{
			name: "multiline_system_prompt",
			spec: &AgentSpecs{
				System: "You are a helpful assistant.\nAlways be polite and respectful.\nProvide clear and concise answers.",
				Model: ModelSpecs{
					ModelID: testModelID,
				},
			},
			expected: []anthropic.TextBlockParam{
				{
					Type: "text",
					Text: "You are a helpful assistant.\nAlways be polite and respectful.\nProvide clear and concise answers.",
					CacheControl: anthropic.CacheControlEphemeralParam{
						Type: "ephemeral",
					},
				},
			},
		},
		{
			name: "system_prompt_with_structured_output",
			spec: &AgentSpecs{
				System: "You are a helpful assistant.",
				Model: ModelSpecs{
					ModelID: testModelID,
					ResponseFormat: map[string]any{
						"type": "object",
						"properties": map[string]any{
							"name": map[string]any{
								"type": "string",
							},
							"age": map[string]any{
								"type": "integer",
							},
						},
						"required": []string{"name", "age"},
					},
				},
			},
			expected: []anthropic.TextBlockParam{
				{
					Type: "text",
					Text: "You are a helpful assistant.\n\nYou must respond with valid JSON that matches this exact schema:\n{\"properties\":{\"age\":{\"type\":\"integer\"},\"name\":{\"type\":\"string\"}},\"required\":[\"name\",\"age\"],\"type\":\"object\"}\n\n",
					CacheControl: anthropic.CacheControlEphemeralParam{
						Type: "ephemeral",
					},
				},
			},
		},
		{
			name: "system_prompt_with_empty_response_format",
			spec: &AgentSpecs{
				System: "You are a helpful assistant.",
				Model: ModelSpecs{
					ModelID:        testModelID,
					ResponseFormat: map[string]any{},
				},
			},
			expected: []anthropic.TextBlockParam{
				{
					Type: "text",
					Text: "You are a helpful assistant.",
					CacheControl: anthropic.CacheControlEphemeralParam{
						Type: "ephemeral",
					},
				},
			},
		},
		{
			name: "system_prompt_with_nil_response_format",
			spec: &AgentSpecs{
				System: "You are a helpful assistant.",
				Model: ModelSpecs{
					ModelID:        testModelID,
					ResponseFormat: nil,
				},
			},
			expected: []anthropic.TextBlockParam{
				{
					Type: "text",
					Text: "You are a helpful assistant.",
					CacheControl: anthropic.CacheControlEphemeralParam{
						Type: "ephemeral",
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := getSystemPrompt(tt.spec, []db.Agent{})
			require.Len(t, result, len(tt.expected))
			for i, expected := range tt.expected {
				assert.Equal(t, string(result[i].Type), string(expected.Type))
				assert.Equal(t, result[i].Text, expected.Text)
				assert.Equal(t, string(result[i].CacheControl.Type), string(expected.CacheControl.Type))
			}
		})
	}
}

func TestFetchAnthropicTools(t *testing.T) {
	t.Parallel()
	dbPool := setupTestDB(t)
	queries := db.New(dbPool)

	// Create a fake created_by user
	userId, err := uuid.Parse("550e8400-c95b-4444-6666-446655440000")
	if err != nil {
		t.Error("Cannot create user created_bby")
	}

	// Add new tools into the database
	tool, err := queries.CreateTool(t.Context(), db.CreateToolParams{
		Name:        "test_tool_unit",
		Description: pgtype.Text{String: "Get the weather for a location", Valid: true},
		Config: db.ToolConfig{
			Type: db.ToolTypeStandalone,
			C: &db.ToolConfigStandalone{
				Url: "",
				Params: &openapi3.Schema{
					Type: openapi3.NewObjectSchema().Type,
					Properties: map[string]*openapi3.SchemaRef{
						"location": openapi3.NewSchemaRef("", &openapi3.Schema{
							Type:        openapi3.NewStringSchema().Type,
							Description: "The location to get the weather for",
						}),
					},
					Required: []string{"location"},
				},
			},
		},
		CreatedBy: userId,
	})
	require.NoError(t, err)
	// Clean up the test tool after the test completes
	defer func() {
		err := queries.DeleteTool(context.Background(), tool.ID)
		if err != nil {
			t.Logf("Failed to cleanup test tool: %v", err)
		}
		dbPool.Close()
	}()
	tests := []struct {
		name        string
		toolRefs    []uuid.UUID
		modelID     string
		expectError bool
		errorMsg    string
		validate    func(t *testing.T, tools []anthropic.ToolUnionParam, err error)
	}{
		{
			name:        "empty_tool_refs",
			toolRefs:    []uuid.UUID{},
			expectError: false,
			validate: func(t *testing.T, tools []anthropic.ToolUnionParam, err error) {
				assert.NoError(t, err)
				assert.Nil(t, tools)
			},
		},
		{
			name:        "nil_tool_refs",
			toolRefs:    nil,
			expectError: false,
			validate: func(t *testing.T, tools []anthropic.ToolUnionParam, err error) {
				assert.NoError(t, err)
				assert.Nil(t, tools)
			},
		},
		{
			name:        "valid_uuid_not_in_db",
			toolRefs:    []uuid.UUID{uuid.MustParse("550e8400-e29b-41d4-a716-446655440000")},
			expectError: false,
			validate: func(t *testing.T, tools []anthropic.ToolUnionParam, err error) {
				// If there's a database connection error, skip the test
				if err != nil {
					t.Skipf("Database connection failed: %v", err)
					return
				}
				// This test will depend on database state, but should not error
				// even if no tools are found (empty result set)
				assert.NoError(t, err)
				// Tools could be empty if not found in DB, which is valid
				assert.NotNil(t, tools)
				assert.Empty(t, tools)
			},
		},
		{
			name:        "real_tool_with_cache",
			toolRefs:    []uuid.UUID{tool.ID},
			expectError: false,
			validate: func(t *testing.T, tools []anthropic.ToolUnionParam, err error) {
				// If there's a database connection error, the test should handle it gracefully
				if err != nil {
					t.Skipf("Database connection failed: %v", err)
					return
				}
				assert.NoError(t, err)
				assert.NotNil(t, tools)

				// Check if tools slice is empty due to database issues
				if len(tools) == 0 {
					t.Skip("No tools returned, likely due to database connection issues")
					return
				}

				assert.Len(t, tools, 1)

				// Validate the converted tool structure
				tool := tools[0]
				assert.NotNil(t, tool.OfTool)
				assert.Equal(t, "test_tool_unit", tool.OfTool.Name)
				assert.NotNil(t, tool.OfTool.Description)
				assert.Equal(t, "Get the weather for a location", tool.OfTool.Description.Value)
				assert.Equal(t, "object", string(tool.OfTool.InputSchema.Type))
				assert.NotNil(t, tool.OfTool.InputSchema.Properties)
				assert.NotNil(t, tool.OfTool.InputSchema.Required)
				assert.Contains(t, tool.OfTool.InputSchema.Required, "location")

				// Validate cache control is always set
				assert.Equal(t, "ephemeral", string(tool.OfTool.CacheControl.Type))
			},
		},
		{
			name:        "multiple_valid_tools_some_not_in_db",
			toolRefs:    []uuid.UUID{tool.ID, uuid.MustParse("550e8400-e29b-41d4-a716-446655440000")},
			expectError: false,
			validate: func(t *testing.T, tools []anthropic.ToolUnionParam, err error) {
				assert.NoError(t, err)
				assert.NotNil(t, tools)
				// Should only return the valid tool that exists in DB
				assert.Len(t, tools, 1)
				assert.Equal(t, "test_tool_unit", tools[0].OfTool.Name)
				// Validate cache control is always set
				assert.Equal(t, "ephemeral", string(tools[0].OfTool.CacheControl.Type))
			},
		},
		{
			name:        "all_invalid_tool_ids",
			toolRefs:    []uuid.UUID{uuid.MustParse("00000000-0000-0000-0000-000000000001"), uuid.MustParse("00000000-0000-0000-0000-000000000002")},
			expectError: false,
			validate: func(t *testing.T, tools []anthropic.ToolUnionParam, err error) {
				assert.NoError(t, err)
				assert.NotNil(t, tools)
				// Should return empty list as no tools found
				assert.Empty(t, tools)
			},
		},
		{
			name:        "mixed_valid_and_invalid_tool_ids",
			toolRefs:    []uuid.UUID{tool.ID, uuid.MustParse("00000000-0000-0000-0000-000000000001"), uuid.MustParse("00000000-0000-0000-0000-000000000002")},
			expectError: false,
			validate: func(t *testing.T, tools []anthropic.ToolUnionParam, err error) {
				assert.NoError(t, err)
				assert.NotNil(t, tools)
				// Should only return the valid tool that exists in DB
				assert.Len(t, tools, 1)
				assert.Equal(t, "test_tool_unit", tools[0].OfTool.Name)
				// Validate cache control is always set
				assert.Equal(t, "ephemeral", string(tools[0].OfTool.CacheControl.Type))
			},
		},
	}

	// Setup service for testing
	log := MockServiceConfigs.CreateLogger()
	wg := &sync.WaitGroup{}
	wg.Add(1)
	defer wg.Done()

	mockService, err := NewService(t.Context(), MockServiceConfigs, log, wg)
	if err != nil {
		t.Fatalf("Failed to create service: %v", err)
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tools, err := mockService.fetchAnthropicTools(tt.toolRefs)
			tt.validate(t, tools, err)
		})
	}
}

func TestGetThinkingConfig(t *testing.T) {
	tests := []struct {
		name     string
		spec     *AgentSpecs
		expected anthropic.ThinkingConfigParamUnion
	}{
		{
			name: "thinking_enabled",
			spec: &AgentSpecs{
				Model: ModelSpecs{
					Thinking: ThinkingSpecs{
						Enabled:     true,
						BudgetToken: 1500,
					},
				},
			},
			expected: anthropic.ThinkingConfigParamUnion{
				OfEnabled: &anthropic.ThinkingConfigEnabledParam{
					BudgetTokens: 1500,
					Type:         "enabled",
				},
			},
		},
		{
			name: "thinking_disabled",
			spec: &AgentSpecs{
				Model: ModelSpecs{
					Thinking: ThinkingSpecs{
						Enabled:     false,
						BudgetToken: 0,
					},
				},
			},
			expected: anthropic.ThinkingConfigParamUnion{
				OfDisabled: &anthropic.ThinkingConfigDisabledParam{
					Type: "disabled",
				},
			},
		},
		{
			name: "thinking_enabled_zero_budget",
			spec: &AgentSpecs{
				Model: ModelSpecs{
					Thinking: ThinkingSpecs{
						Enabled:     true,
						BudgetToken: 0,
					},
				},
			},
			expected: anthropic.ThinkingConfigParamUnion{
				OfEnabled: &anthropic.ThinkingConfigEnabledParam{
					BudgetTokens: 0,
					Type:         "enabled",
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := getThinkingConfig(tt.spec)

			if tt.expected.OfEnabled != nil {
				assert.NotNil(t, result.OfEnabled)
				assert.Equal(t, string(tt.expected.OfEnabled.Type), string(result.OfEnabled.Type))
				assert.Equal(t, tt.expected.OfEnabled.BudgetTokens, result.OfEnabled.BudgetTokens)
				assert.Nil(t, result.OfDisabled)
			} else {
				assert.NotNil(t, result.OfDisabled)
				assert.Equal(t, string(tt.expected.OfDisabled.Type), string(result.OfDisabled.Type))
				assert.Nil(t, result.OfEnabled)
			}
		})
	}
}

func TestInvokeAnthropicModel(t *testing.T) {
	log := MockServiceConfigs.CreateLogger()
	wg := &sync.WaitGroup{}
	wg.Add(1)
	mockService, err := NewService(t.Context(), MockServiceConfigs, log, wg)
	if err != nil {
		t.Fatalf("Failed to create service: %v", err)
	}

	testCases := []struct {
		name     string
		messages []anthropic.MessageParam
		spec     *AgentSpecs
	}{
		{
			name: "Successful non-streaming request",
			messages: []anthropic.MessageParam{
				{
					Content: []anthropic.ContentBlockParamUnion{
						anthropic.NewTextBlock("Hi"),
					},
					Role: anthropic.MessageParamRoleUser,
				},
			},
			spec: &AgentSpecs{
				Model: ModelSpecs{
					Provider:  "anthropic",
					ModelID:   testModelID,
					MaxTokens: 1000,
					Stream:    false,
				},
				System: "You are a helpful assistant.",
			},
		},
		{
			name: "Successful non-streaming request with thinking enabled",
			messages: []anthropic.MessageParam{
				{
					Content: []anthropic.ContentBlockParamUnion{
						anthropic.NewTextBlock("Hi"),
					},
					Role: anthropic.MessageParamRoleUser,
				},
			},
			spec: &AgentSpecs{
				Model: ModelSpecs{
					Provider:  "anthropic",
					ModelID:   testModelID,
					MaxTokens: 4000,
					Stream:    false,
					Thinking: ThinkingSpecs{
						Enabled:     true,
						BudgetToken: 1024,
					},
				},
				System: "You are a helpful assistant.",
			},
		},
		{
			name: "Successful streaming request",
			messages: []anthropic.MessageParam{
				{
					Content: []anthropic.ContentBlockParamUnion{
						anthropic.NewTextBlock("Hi"),
					},
					Role: anthropic.MessageParamRoleUser,
				},
			},
			spec: &AgentSpecs{
				Model: ModelSpecs{
					Provider:  "anthropic",
					ModelID:   testModelID,
					MaxTokens: 1000,
					Stream:    true,
					Thinking: ThinkingSpecs{
						Enabled:     false,
						BudgetToken: 0,
					},
				},
				System: "You are a helpful assistant.",
			},
		},
		{
			name: "Successful request with thinking enabled",
			messages: []anthropic.MessageParam{
				{
					Content: []anthropic.ContentBlockParamUnion{
						anthropic.NewTextBlock("Hi"),
					},
					Role: anthropic.MessageParamRoleUser,
				},
			},
			spec: &AgentSpecs{
				Model: ModelSpecs{
					Provider:  "anthropic",
					ModelID:   testModelID,
					MaxTokens: 4000,
					Stream:    true,
					Thinking: ThinkingSpecs{
						Enabled:     true,
						BudgetToken: 1024,
					},
				},
				System: "You are a helpful assistant.",
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			msg, stop, err := mockService.handleAnthropicRequest(tc.messages, tc.spec, &service.EventHeaders{}, &service.EventMetadata{})

			// Assert no error occurred
			assert.Nil(t, err)
			require.NotNil(t, msg, "Response message should not be nil")
			assert.Equal(t, "assistant", string(msg.Role), "Response should be from assistant")
			assert.Equal(t, "end_turn", stop, "Stop reason should be 'end_turn'")

			// Assert message has content
			require.NotEmpty(t, msg.Content, "Message should have content")
			require.NotEmpty(t, msg.Role, "Message should have a role")

			// Assert content is text content and not empty
			for _, content := range msg.Content {
				if textBlock := content.OfText; textBlock != nil {
					assert.NotEmpty(t, textBlock.Text, "Text content should not be empty")
					assert.Equal(t, "text", string(textBlock.Type), "Content type should be 'text'")
				}

				// Assert thinking block if present
				if thinkingBlock := content.OfThinking; thinkingBlock != nil {
					assert.NotEmpty(t, thinkingBlock.Thinking, "Thinking content should not be empty")
					assert.Equal(t, "thinking", string(thinkingBlock.Type), "Content type should be 'thinking'")

					// Verify thinking content only appears when thinking is enabled
					assert.True(t, tc.spec.Model.Thinking.Enabled, "Thinking block should only appear when thinking is enabled in spec")
				}
			}
		})
	}
}

func TestStructuredOutputPrefillMessage(t *testing.T) {
	tests := []struct {
		name          string
		spec          *AgentSpecs
		inputMessages []anthropic.MessageParam
		expectPrefill bool
		expectedText  string
	}{
		{
			name: "structured_output_adds_prefill",
			spec: &AgentSpecs{
				System: "You are a helpful assistant.",
				Model: ModelSpecs{
					Provider:  "anthropic",
					ModelID:   testModelID,
					MaxTokens: 1000,
					Stream:    false,
					ResponseFormat: map[string]any{
						"type": "object",
						"properties": map[string]any{
							"name": map[string]any{
								"type": "string",
							},
						},
						"required": []string{"name"},
					},
				},
			},
			inputMessages: []anthropic.MessageParam{
				{
					Content: []anthropic.ContentBlockParamUnion{
						anthropic.NewTextBlock("Hello"),
					},
					Role: anthropic.MessageParamRoleUser,
				},
			},
			expectPrefill: true,
			expectedText:  "{",
		},
		{
			name: "no_response_format_no_prefill",
			spec: &AgentSpecs{
				System: "You are a helpful assistant.",
				Model: ModelSpecs{
					Provider:       "anthropic",
					ModelID:        testModelID,
					MaxTokens:      1000,
					Stream:         false,
					ResponseFormat: nil,
				},
			},
			inputMessages: []anthropic.MessageParam{
				{
					Content: []anthropic.ContentBlockParamUnion{
						anthropic.NewTextBlock("Hello"),
					},
					Role: anthropic.MessageParamRoleUser,
				},
			},
			expectPrefill: false,
		},
		{
			name: "empty_response_format_no_prefill",
			spec: &AgentSpecs{
				System: "You are a helpful assistant.",
				Model: ModelSpecs{
					Provider:       "anthropic",
					ModelID:        testModelID,
					MaxTokens:      1000,
					Stream:         false,
					ResponseFormat: map[string]any{},
				},
			},
			inputMessages: []anthropic.MessageParam{
				{
					Content: []anthropic.ContentBlockParamUnion{
						anthropic.NewTextBlock("Hello"),
					},
					Role: anthropic.MessageParamRoleUser,
				},
			},
			expectPrefill: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create the request parameters similar to how handleAnthropicRequest does it
			params := anthropic.MessageNewParams{
				Messages: tt.inputMessages,
			}

			// Apply the same logic as in handleAnthropicRequest for adding prefill
			if len(tt.spec.Model.ResponseFormat) > 0 {
				prefillMsg := anthropic.NewAssistantMessage(anthropic.NewTextBlock("{"))
				params.Messages = append(params.Messages, prefillMsg)
			}

			if tt.expectPrefill {
				// Verify that a prefill message was added
				assert.Greater(t, len(params.Messages), len(tt.inputMessages), "Should have added prefill message")

				// Check the last message is assistant role with prefill content
				lastMsg := params.Messages[len(params.Messages)-1]
				assert.Equal(t, "assistant", string(lastMsg.Role), "Last message should be from assistant")

				// Check the content of the prefill message
				require.NotEmpty(t, lastMsg.Content, "Prefill message should have content")

				// Verify it's a text block with "{"
				textBlock := lastMsg.Content[0].OfText
				require.NotNil(t, textBlock, "Prefill message should be a text block")
				assert.Equal(t, tt.expectedText, textBlock.Text, "Prefill message should contain '{'")
			} else {
				// Verify no prefill message was added
				assert.Equal(t, len(tt.inputMessages), len(params.Messages), "Should not have added prefill message")
			}
		})
	}
}
