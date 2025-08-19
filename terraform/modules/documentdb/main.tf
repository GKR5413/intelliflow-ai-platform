# DocumentDB Placeholder
resource "aws_docdb_cluster" "main" {
  cluster_identifier = "${var.project_name}-${var.environment}-docdb"
  engine             = "docdb"
  master_username    = "docdb"
  master_password    = "password123"
  
  tags = var.tags
}
