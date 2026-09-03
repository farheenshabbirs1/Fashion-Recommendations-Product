variable "location" {
  description = "Azure region to deploy into."
  type        = string
  default     = "eastus"
}

variable "resource_group_name" {
  type    = string
  default = "fashion-rag-rg"
}

variable "cluster_name" {
  type    = string
  default = "fashion-rag-aks"
}

variable "acr_name" {
  description = "Globally unique; alphanumeric only."
  type        = string
  default     = "fashionragacr"
}

variable "node_count" {
  type    = number
  default = 2
}

variable "node_vm_size" {
  type    = string
  default = "Standard_B2s"
}

variable "kubernetes_version" {
  description = "Leave null to use whatever AKS currently recommends as default."
  type        = string
  default     = null
}
