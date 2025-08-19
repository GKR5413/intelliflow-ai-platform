# Global Variables
variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "intelliflow"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
  
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

# VPC Configuration
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

# EKS Configuration
variable "eks_cluster_version" {
  description = "Kubernetes version for EKS cluster"
  type        = string
  default     = "1.28"
}

variable "eks_node_groups" {
  description = "EKS node group configurations"
  type = map(object({
    instance_types = list(string)
    capacity_type  = string
    scaling_config = object({
      desired_size = number
      max_size     = number
      min_size     = number
    })
    update_config = object({
      max_unavailable = number
    })
  }))
  default = {
    main = {
      instance_types = ["t3.medium"]
      capacity_type  = "ON_DEMAND"
      scaling_config = {
        desired_size = 3
        max_size     = 10
        min_size     = 1
      }
      update_config = {
        max_unavailable = 1
      }
    }
    spot = {
      instance_types = ["t3.medium", "t3.large"]
      capacity_type  = "SPOT"
      scaling_config = {
        desired_size = 2
        max_size     = 8
        min_size     = 0
      }
      update_config = {
        max_unavailable = 1
      }
    }
  }
}

# RDS Configuration
variable "rds_databases" {
  description = "RDS database configurations"
  type = map(object({
    engine                = string
    engine_version        = string
    instance_class        = string
    allocated_storage     = number
    max_allocated_storage = number
    storage_encrypted     = bool
    multi_az              = bool
    backup_retention_period = number
    backup_window         = string
    maintenance_window    = string
    deletion_protection   = bool
  }))
  default = {
    main = {
      engine                = "postgres"
      engine_version        = "15.4"
      instance_class        = "db.t3.micro"
      allocated_storage     = 20
      max_allocated_storage = 100
      storage_encrypted     = true
      multi_az              = false
      backup_retention_period = 7
      backup_window         = "03:00-04:00"
      maintenance_window    = "Mon:04:00-Mon:05:00"
      deletion_protection   = false
    }
  }
}

# ElastiCache Configuration
variable "redis_clusters" {
  description = "Redis cluster configurations"
  type = map(object({
    node_type                = string
    num_cache_nodes          = number
    parameter_group_name     = string
    engine_version           = string
    port                     = number
    maintenance_window       = string
    snapshot_retention_limit = number
    snapshot_window          = string
  }))
  default = {
    main = {
      node_type                = "cache.t3.micro"
      num_cache_nodes          = 1
      parameter_group_name     = "default.redis7"
      engine_version           = "7.0"
      port                     = 6379
      maintenance_window       = "sun:05:00-sun:06:00"
      snapshot_retention_limit = 5
      snapshot_window          = "03:00-05:00"
    }
  }
}

# S3 Configuration
variable "s3_buckets" {
  description = "S3 bucket configurations"
  type = map(object({
    versioning_enabled = bool
    encryption_enabled = bool
    lifecycle_enabled  = bool
    public_access_block = bool
  }))
  default = {
    app-storage = {
      versioning_enabled = true
      encryption_enabled = true
      lifecycle_enabled  = true
      public_access_block = true
    }
    analytics-data = {
      versioning_enabled = true
      encryption_enabled = true
      lifecycle_enabled  = true
      public_access_block = true
    }
    fraud-models = {
      versioning_enabled = true
      encryption_enabled = true
      lifecycle_enabled  = false
      public_access_block = true
    }
    notification-assets = {
      versioning_enabled = false
      encryption_enabled = true
      lifecycle_enabled  = true
      public_access_block = true
    }
  }
}

# MSK Configuration
variable "msk_kafka_version" {
  description = "Kafka version for MSK"
  type        = string
  default     = "2.8.1"
}

variable "msk_instance_type" {
  description = "Instance type for MSK brokers"
  type        = string
  default     = "kafka.t3.small"
}

# DocumentDB Configuration
variable "documentdb_cluster_size" {
  description = "Number of DocumentDB instances"
  type        = number
  default     = 2
}

variable "documentdb_instance_class" {
  description = "Instance class for DocumentDB"
  type        = string
  default     = "db.t3.medium"
}
