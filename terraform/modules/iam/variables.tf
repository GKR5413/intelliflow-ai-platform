variable "project_name" { type = string }
variable "environment" { type = string }
variable "eks_cluster_name" { type = string }
variable "s3_bucket_names" { type = map(string) }
variable "tags" { type = map(string); default = {} }
