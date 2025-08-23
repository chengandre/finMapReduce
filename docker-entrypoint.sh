#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting MapReduce QA WebApp...${NC}"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check environment variables
check_env_vars() {
    echo -e "${BLUE}📋 Checking environment configuration...${NC}"
    
    # Check if at least one API key is set
    if [[ -z "$OPENAI_API_KEY" && -z "$OPENROUTER_API_KEY" && -z "$SELF_OPENAI_API_KEY" ]]; then
        echo -e "${RED}❌ Error: No API keys found!${NC}"
        echo -e "${YELLOW}   Please set at least one of:${NC}"
        echo "   - OPENAI_API_KEY"
        echo "   - OPENROUTER_API_KEY"
        echo "   - SELF_OPENAI_API_KEY"
        echo ""
        echo -e "${YELLOW}   You can set these in your .env file${NC}"
        exit 1
    fi
    
    # Log which API keys are available (without showing the actual keys)
    if [[ -n "$OPENAI_API_KEY" ]]; then
        echo -e "${GREEN}✅ OpenAI API key configured${NC}"
    fi
    if [[ -n "$OPENROUTER_API_KEY" ]]; then
        echo -e "${GREEN}✅ OpenRouter API key configured${NC}"
    fi
    if [[ -n "$SELF_OPENAI_API_KEY" ]]; then
        echo -e "${GREEN}✅ Self OpenAI API key configured${NC}"
    fi
}

# Function to create necessary directories
create_directories() {
    echo -e "${BLUE}📁 Creating necessary directories...${NC}"
    
    # Create directories with proper permissions
    mkdir -p /app/marker /app/pdf_cache /app/webapp/backend/prompts_log /tmp/webapp_uploads
    
    echo -e "${GREEN}✅ Directories created${NC}"
}

# Main execution
main() {
    echo -e "${BLUE}🐳 MapReduce QA WebApp Docker Container${NC}"
    echo "=================================================="
    
    # Run checks
    check_env_vars
    create_directories
    
    # Set default values
    export HOST=${HOST:-0.0.0.0}
    export PORT=${PORT:-8000}
    
    echo ""
    echo -e "${GREEN}🎉 Configuration complete!${NC}"
    echo -e "${BLUE}📱 Starting server on ${HOST}:${PORT}${NC}"
    echo -e "${BLUE}🌐 Frontend: http://localhost:${PORT}${NC}"
    echo -e "${BLUE}📚 API Docs: http://localhost:${PORT}/docs${NC}"
    echo ""
    
    # Execute the command passed to the container
    exec "$@"
}

# Run main function
main "$@"