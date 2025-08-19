# Basic IAM roles placeholder
resource "aws_iam_role" "eks_service_role" {
  name = "${var.project_name}-${var.environment}-eks-service-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = { Service = "eks.amazonaws.com" }
    }]
  })
  
  tags = var.tags
}
