output "redis_endpoint" {
  description = "Redis cluster endpoint"
  value       = values(aws_elasticache_cluster.redis)[0].cache_nodes[0].address
}

output "redis_port" {
  description = "Redis cluster port"
  value       = values(aws_elasticache_cluster.redis)[0].cache_nodes[0].port
}
