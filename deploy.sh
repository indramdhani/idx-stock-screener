#!/bin/bash

# Indonesian Stock Screener v2.0 - Docker Deployment Script
# Enhanced Features & Optimization with Phase 7-8 capabilities
#
# This script handles deployment of the IDX Stock Screener with Docker
# Supports multiple environments: development, staging, production
#
# Usage:
#   ./docker/deploy.sh [environment] [action] [options]
#
# Examples:
#   ./docker/deploy.sh dev up                    # Start development environment
#   ./docker/deploy.sh prod up --build          # Build and start production
#   ./docker/deploy.sh staging down             # Stop staging environment
#   ./docker/deploy.sh prod logs dashboard      # View dashboard logs

set -e

# ==============================================================================
# CONFIGURATION
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Default values
DEFAULT_ENV="dev"
DEFAULT_ACTION="up"
COMPOSE_PROJECT_NAME="idx-screener"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

fatal() {
    error "$1"
    exit 1
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check required dependencies
check_dependencies() {
    log "Checking dependencies..."

    if ! command_exists docker; then
        fatal "Docker is not installed. Please install Docker first."
    fi

    if ! command_exists docker-compose; then
        fatal "Docker Compose is not installed. Please install Docker Compose first."
    fi

    # Check Docker daemon is running
    if ! docker info >/dev/null 2>&1; then
        fatal "Docker daemon is not running. Please start Docker first."
    fi

    success "All dependencies are available"
}

# Get Docker Compose version
get_compose_version() {
    if docker-compose version >/dev/null 2>&1; then
        echo "docker-compose"
    elif docker compose version >/dev/null 2>&1; then
        echo "docker compose"
    else
        fatal "No compatible Docker Compose found"
    fi
}

# ==============================================================================
# ENVIRONMENT CONFIGURATION
# ==============================================================================

setup_environment() {
    local env=$1

    log "Setting up environment: ${env}"

    # Create necessary directories
    mkdir -p logs data models exports backups
    mkdir -p docker/data/{redis,postgres,prometheus,grafana}

    # Set environment file
    case $env in
        "dev"|"development")
            ENV_FILE=".env.dev"
            COMPOSE_FILES="-f docker-compose.yml -f docker-compose.dev.yml"
            PROFILES="--profile dev"
            ;;
        "staging")
            ENV_FILE=".env.staging"
            COMPOSE_FILES="-f docker-compose.yml -f docker-compose.staging.yml"
            PROFILES="--profile staging --profile database"
            ;;
        "prod"|"production")
            ENV_FILE=".env.prod"
            COMPOSE_FILES="-f docker-compose.yml -f docker-compose.prod.yml"
            PROFILES="--profile production --profile database --profile monitoring"
            ;;
        *)
            warning "Unknown environment: ${env}. Using development configuration."
            ENV_FILE=".env.dev"
            COMPOSE_FILES="-f docker-compose.yml"
            PROFILES=""
            ;;
    esac

    # Check if environment file exists
    if [[ ! -f "$ENV_FILE" ]]; then
        if [[ -f ".env.example" ]]; then
            warning "Environment file ${ENV_FILE} not found. Creating from template..."
            cp .env.example "$ENV_FILE"
            warning "Please edit ${ENV_FILE} with your configuration before proceeding."
        else
            fatal "Environment file ${ENV_FILE} not found and no template available."
        fi
    fi

    # Export environment variables
    export COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME}-${env}"
    export COMPOSE_FILE="${COMPOSE_FILES}"
    export ENV_FILE

    success "Environment ${env} configured"
}

# ==============================================================================
# DOCKER OPERATIONS
# ==============================================================================

# Build Docker images
build_images() {
    local rebuild=$1

    log "Building Docker images..."

    local build_args=""
    if [[ "$rebuild" == "true" ]]; then
        build_args="--no-cache --pull"
    fi

    $(get_compose_version) --env-file="$ENV_FILE" build $build_args

    success "Docker images built successfully"
}

# Start services
start_services() {
    local background=$1

    log "Starting IDX Stock Screener services..."

    local up_args="-d"
    if [[ "$background" == "false" ]]; then
        up_args=""
    fi

    $(get_compose_version) --env-file="$ENV_FILE" $PROFILES up $up_args

    if [[ "$background" != "false" ]]; then
        log "Services started in background. Use 'logs' command to view output."
        show_status
    fi

    success "Services started successfully"
}

# Stop services
stop_services() {
    log "Stopping IDX Stock Screener services..."

    $(get_compose_version) --env-file="$ENV_FILE" down

    success "Services stopped successfully"
}

# Show service status
show_status() {
    log "Service Status:"
    $(get_compose_version) --env-file="$ENV_FILE" ps

    log "Health Status:"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep idx-
}

# Show service logs
show_logs() {
    local service=$1
    local follow=${2:-false}

    local log_args=""
    if [[ "$follow" == "true" ]]; then
        log_args="-f"
    fi

    if [[ -n "$service" ]]; then
        log "Showing logs for service: ${service}"
        $(get_compose_version) --env-file="$ENV_FILE" logs $log_args "$service"
    else
        log "Showing logs for all services:"
        $(get_compose_version) --env-file="$ENV_FILE" logs $log_args
    fi
}

# Execute command in container
exec_command() {
    local service=$1
    shift
    local command="$*"

    log "Executing command in ${service}: ${command}"
    $(get_compose_version) --env-file="$ENV_FILE" exec "$service" $command
}

# Scale services
scale_services() {
    local service=$1
    local replicas=$2

    log "Scaling ${service} to ${replicas} replicas..."
    $(get_compose_version) --env-file="$ENV_FILE" up -d --scale "${service}=${replicas}"

    success "Service ${service} scaled to ${replicas} replicas"
}

# ==============================================================================
# MAINTENANCE OPERATIONS
# ==============================================================================

# Update services
update_services() {
    log "Updating IDX Stock Screener services..."

    # Pull latest images
    $(get_compose_version) --env-file="$ENV_FILE" pull

    # Rebuild and restart
    $(get_compose_version) --env-file="$ENV_FILE" up -d --build

    success "Services updated successfully"
}

# Backup data
backup_data() {
    local backup_dir="backups/$(date +%Y%m%d_%H%M%S)"

    log "Creating backup in ${backup_dir}..."
    mkdir -p "$backup_dir"

    # Backup database
    if $(get_compose_version) --env-file="$ENV_FILE" ps postgres >/dev/null 2>&1; then
        log "Backing up PostgreSQL database..."
        $(get_compose_version) --env-file="$ENV_FILE" exec -T postgres pg_dump -U idx_user idx_screener > "${backup_dir}/postgres_backup.sql"
    fi

    # Backup Redis data
    if $(get_compose_version) --env-file="$ENV_FILE" ps redis >/dev/null 2>&1; then
        log "Backing up Redis data..."
        cp -r docker/data/redis "${backup_dir}/redis_backup" 2>/dev/null || true
    fi

    # Backup application data
    log "Backing up application data..."
    cp -r data "${backup_dir}/app_data" 2>/dev/null || true
    cp -r models "${backup_dir}/models" 2>/dev/null || true
    cp -r logs "${backup_dir}/logs" 2>/dev/null || true

    success "Backup created: ${backup_dir}"
}

# Clean up old resources
cleanup() {
    log "Cleaning up old resources..."

    # Remove stopped containers
    docker container prune -f

    # Remove unused images
    docker image prune -f

    # Remove unused volumes (with confirmation)
    echo -n "Remove unused volumes? [y/N]: "
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        docker volume prune -f
    fi

    # Remove unused networks
    docker network prune -f

    success "Cleanup completed"
}

# ==============================================================================
# MONITORING OPERATIONS
# ==============================================================================

# Health check
health_check() {
    log "Performing health check..."

    local failed=0

    # Check main dashboard
    if ! curl -sf http://localhost:5000/api/status >/dev/null 2>&1; then
        error "Dashboard health check failed"
        ((failed++))
    else
        success "Dashboard is healthy"
    fi

    # Check Redis
    if $(get_compose_version) --env-file="$ENV_FILE" ps redis >/dev/null 2>&1; then
        if ! $(get_compose_version) --env-file="$ENV_FILE" exec -T redis redis-cli ping >/dev/null 2>&1; then
            error "Redis health check failed"
            ((failed++))
        else
            success "Redis is healthy"
        fi
    fi

    # Check PostgreSQL
    if $(get_compose_version) --env-file="$ENV_FILE" ps postgres >/dev/null 2>&1; then
        if ! $(get_compose_version) --env-file="$ENV_FILE" exec -T postgres pg_isready -U idx_user >/dev/null 2>&1; then
            error "PostgreSQL health check failed"
            ((failed++))
        else
            success "PostgreSQL is healthy"
        fi
    fi

    if [[ $failed -eq 0 ]]; then
        success "All services are healthy"
        return 0
    else
        error "${failed} service(s) failed health check"
        return 1
    fi
}

# Show resource usage
show_resources() {
    log "Resource Usage:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}\t{{.BlockIO}}" | grep idx-
}

# ==============================================================================
# MAIN SCRIPT LOGIC
# ==============================================================================

show_help() {
    cat << EOF
Indonesian Stock Screener v2.0 - Docker Deployment Script

Usage: $0 [environment] [action] [options]

Environments:
  dev, development    Development environment (default)
  staging            Staging environment with database
  prod, production   Production environment with all services

Actions:
  up                 Start services (default)
  down               Stop services
  build              Build Docker images
  rebuild            Rebuild Docker images (no cache)
  restart            Restart all services
  status             Show service status
  logs [service]     Show logs (optionally for specific service)
  follow [service]   Follow logs in real-time
  exec service cmd   Execute command in service container
  scale service N    Scale service to N replicas
  update             Update services to latest version
  backup             Create backup of data
  cleanup            Clean up old Docker resources
  health             Run health checks
  resources          Show resource usage
  validate           Validate Phase 7-8 implementation

Options:
  --build            Force rebuild during up
  --follow           Follow logs in real-time
  --no-deps          Don't start dependent services
  --help, -h         Show this help message

Examples:
  $0 dev up --build                    # Build and start development
  $0 prod up                          # Start production environment
  $0 staging logs dashboard           # Show dashboard logs in staging
  $0 prod exec idx-dashboard bash     # Access dashboard container
  $0 dev health                       # Check service health
  $0 prod backup                      # Create production backup

For more information, visit: https://github.com/your-repo/idx-stock-screener
EOF
}

main() {
    local environment="$DEFAULT_ENV"
    local action="$DEFAULT_ACTION"
    local options=()

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            dev|development|staging|prod|production)
                environment=$1
                shift
                ;;
            up|down|build|rebuild|restart|status|logs|follow|exec|scale|update|backup|cleanup|health|resources|validate)
                action=$1
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                options+=("$1")
                shift
                ;;
        esac
    done

    # Initialize
    check_dependencies
    setup_environment "$environment"

    # Execute action
    case $action in
        "up")
            local build_flag=false
            local background=true
            for opt in "${options[@]}"; do
                [[ "$opt" == "--build" ]] && build_flag=true
                [[ "$opt" == "--no-background" ]] && background=false
            done
            [[ "$build_flag" == "true" ]] && build_images false
            start_services "$background"
            ;;
        "down")
            stop_services
            ;;
        "build")
            build_images false
            ;;
        "rebuild")
            build_images true
            ;;
        "restart")
            stop_services
            start_services true
            ;;
        "status")
            show_status
            ;;
        "logs")
            local follow=false
            for opt in "${options[@]}"; do
                [[ "$opt" == "--follow" ]] && follow=true
            done
            show_logs "${options[0]}" "$follow"
            ;;
        "follow")
            show_logs "${options[0]}" true
            ;;
        "exec")
            if [[ ${#options[@]} -lt 2 ]]; then
                fatal "Usage: $0 $environment exec <service> <command>"
            fi
            exec_command "${options[@]}"
            ;;
        "scale")
            if [[ ${#options[@]} -lt 2 ]]; then
                fatal "Usage: $0 $environment scale <service> <replicas>"
            fi
            scale_services "${options[0]}" "${options[1]}"
            ;;
        "update")
            update_services
            ;;
        "backup")
            backup_data
            ;;
        "cleanup")
            cleanup
            ;;
        "health")
            health_check
            ;;
        "resources")
            show_resources
            ;;
        "validate")
            log "Validating Phase 7-8 implementation..."
            exec_command "idx-dashboard" "python3" "validate_phase7_8.py"
            ;;
        *)
            error "Unknown action: $action"
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
