variable "project_name" {
  description = "Name of the project"
  type        = string
}

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "vpc_id" {
  description = "ID of the VPC"
  type        = string
}

variable "private_subnet_ids" {
  description = "List of private subnet IDs"
  type        = list(string)
}

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
}

variable "tags" {
  description = "A map of tags to assign to the resource"
  type        = map(string)
  default     = {}
}
