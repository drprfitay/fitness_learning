library(cowplot)
library(ggplot2)
library(pheatmap)

root_path = "/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning"
results_path = sprintf("%s/results/", root_path)
base_path = sprintf("%s/data/configuration/", root_path)
figures_path = sprintf("%s/figures/gfp_dataset/esm3", results_path)
evaluation_path = sprintf("%s/gfp_dataset/finetuned_models_evaluated/", results_path)
dir.create(figures_path)

model_name = "esm2_600m_orpo_nll_1_2_3_4"

seq_df = read.csv(paste(base_path, "fixed_unique_gfp_sequence_dataset_full_seq.csv", sep=""))

masks = read.csv(sprintf("%s/esm2_600m_orpo_nll_1_2_3_4/evaluated_sequence_masks_onehot.csv", evaluation_path))
test_fitness = read.csv(sprintf("%s/esm2_600m_orpo_nll_1_2_3_4/test_results.csv", evaluation_path))
train_fitness = read.csv(sprintf("%s/esm2_600m_orpo_nll_1_2_3_4/train_results.csv", evaluation_path))
test_fitness <- cbind(test_fitness, seq_df[(test_fitness[,6] + 1),"num_of_muts"] )

colnames(test_fitness) <- c("#", "mask", "part_fit", "full_fit", "label", "ind", "nm")
colnames(train_fitness) <- c("#", "mask", "part_fit", "full_fit", "label", "ind")


gl <- list()

all_nm = sort(unique(test_fitness$nm))[-1][1:6]
for (nm in all_nm) {
  
  ind = test_fitness$nm == nm
  
  sub_ind = order(test_fitness[ind,"full_fit"], decreasing = T)[1:100]
  g <- 
  ggplot(test_fitness[which(ind)[sub_ind],]) +
    geom_histogram(aes(x=full_fit,
                       color=as.factor(label), 
                       group=as.factor(label), 
                       fill=as.factor(label)),
                   bins=200, alpha=.5, 
                   position="identity") +
  ylab("Frequency") +
  xlab("ESM score") +
  ggtitle(sprintf("Number of mutations: %d", nm)) +
  theme(legend.position = "none") 
  
  
  gl <- append(gl, list(g))
  
}

gl$nrow <- 2
do.call(plot_grid, gl)

