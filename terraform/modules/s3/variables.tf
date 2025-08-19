variable "project_name" {
  description = "Name of the project"
  type        = string
}

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "buckets" {
  description = "S3 bucket configurations"
  type = map(object({
    versioning_enabled = bool
    encryption_enabled = bool
    lifecycle_enabled  = bool
    public_access_block = bool
  }))
}

variable "tags" {
  description = "A map of tags to assign to the resource"
  type        = map(string)
  default     = {}
}
