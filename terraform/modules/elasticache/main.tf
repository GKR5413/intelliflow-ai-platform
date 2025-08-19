# ElastiCache Subnet Group
resource "aws_elasticache_subnet_group" "main" {
  name       = "${var.project_name}-${var.environment}-cache-subnet"
  subnet_ids = var.private_subnet_ids

  tags = var.tags
}

# Security Group for ElastiCache
resource "aws_security_group" "redis" {
  name_prefix = "${var.project_name}-${var.environment}-redis-"
  vpc_id      = var.vpc_id

  ingress {
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = [data.aws_vpc.main.cidr_block]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(var.tags, {
    Name = "${var.project_name}-${var.environment}-redis-sg"
  })

  lifecycle {
    create_before_destroy = true
  }
}

# ElastiCache Redis Clusters
resource "aws_elasticache_cluster" "redis" {
  for_each = var.redis_clusters

  cluster_id           = "${var.project_name}-${var.environment}-${each.key}"
  engine               = "redis"
  node_type            = each.value.node_type
  num_cache_nodes      = each.value.num_cache_nodes
  parameter_group_name = each.value.parameter_group_name
  port                 = each.value.port
  subnet_group_name    = aws_elasticache_subnet_group.main.name
  security_group_ids   = [aws_security_group.redis.id]

  maintenance_window       = each.value.maintenance_window
  snapshot_retention_limit = each.value.snapshot_retention_limit
  snapshot_window          = each.value.snapshot_window

  tags = merge(var.tags, {
    Name = "${var.project_name}-${var.environment}-${each.key}-redis"
  })
}

data "aws_vpc" "main" {
  id = var.vpc_id
}
