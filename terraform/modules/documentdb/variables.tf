variable "project_name" { type = string }
variable "environment" { type = string }
variable "vpc_id" { type = string }
variable "private_subnet_ids" { type = list(string) }
variable "cluster_size" { type = number }
variable "instance_class" { type = string }
variable "tags" { type = map(string); default = {} }
