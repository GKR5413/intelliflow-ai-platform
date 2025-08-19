output "cluster_endpoint" {
  description = "RDS cluster endpoint"
  value       = values(aws_rds_cluster.main)[0].endpoint
}

output "cluster_reader_endpoint" {
  description = "RDS cluster reader endpoint"
  value       = values(aws_rds_cluster.main)[0].reader_endpoint
}

output "cluster_port" {
  description = "RDS cluster port"
  value       = values(aws_rds_cluster.main)[0].port
}

output "cluster_database_name" {
  description = "RDS cluster database name"
  value       = values(aws_rds_cluster.main)[0].database_name
}

output "cluster_master_username" {
  description = "RDS cluster master username"
  value       = values(aws_rds_cluster.main)[0].master_username
  sensitive   = true
}

output "cluster_identifier" {
  description = "RDS cluster identifier"
  value       = values(aws_rds_cluster.main)[0].cluster_identifier
}

output "security_group_id" {
  description = "Security group ID for RDS"
  value       = aws_security_group.rds.id
}

output "secret_arn" {
  description = "ARN of the secret containing RDS credentials"
  value       = aws_secretsmanager_secret.rds_password.arn
}
