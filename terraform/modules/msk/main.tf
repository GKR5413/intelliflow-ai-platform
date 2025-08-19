# Placeholder for MSK module - basic implementation
resource "aws_msk_cluster" "main" {
  cluster_name           = "${var.project_name}-${var.environment}-kafka"
  kafka_version          = var.kafka_version
  number_of_broker_nodes = 3

  broker_node_group_info {
    instance_type   = var.instance_type
    ebs_volume_size = 100
    client_subnets  = var.private_subnet_ids
  }

  tags = var.tags
}
