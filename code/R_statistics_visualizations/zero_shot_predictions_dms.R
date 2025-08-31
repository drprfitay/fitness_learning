library(ggplot2)
library(plyr)
library(cowplot)
library(pheatmap)

results_path = "/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning/results"
root_dms_path = sprintf("%s/zero_shot_predictions_dms/", results_path)
figures_general_path = sprintf("%s/figures", results_path)
figures_path = sprintf("%s/zero_shot_predictions_dms/", figures_general_path)

dir.create(figures_general_path)
dir.create(figures_path)


models_to_analyze = ""

dms_results_to_analyse = c("esm3/pretraining_baseline_1f_fitness_prediction",
                           "esm2/esm1_t34_670M_UR100/pretraining_baseline_fitness_prediction",
                           "esm2/esm1b_t33_650M_UR50S/pretraining_baseline_fitness_prediction",   
                           "esm2/esm1v_t33_650M_UR90S_1/pretraining_baseline_fitness_prediction",
                           "esm2/esm2_t33_650M_UR50D/pretraining_baseline_fitness_prediction",
                           "esm2/esm2_t36_3B_UR50D/pretraining_baseline_fitness_prediction")

names(dms_results_to_analyse) <- c("esm3",
                                   "esm1_t34_670M_UR100",
                                   "esm1b_t33_650M_UR50S",
                                   "esm1v_t33_650M_UR90S_1",
                                   "esm2_t33_650M_UR50D",
                                   "esm2_t36_3B_UR50D")


#example_dms = sample(unique(full_values$DMS), 3)
example_dms = c("MK01_HUMAN_Brenan2016_DOX_Average", 
                "AMIE_PSEAE_Wrenbeck2017_isobutyramide_normalized_fitness",
                "UBC9_HUMAN_Roth2017_screenscore")

all_average_results = c()

for (model_name in names(dms_results_to_analyse)) {
  res = dms_results_to_analyse[model_name]
  fitness_results_df = read.csv(sprintf("%s/%s.csv", root_dms_path, res))
  full_values = read.csv(sprintf("%s/%s_full_values.csv", root_dms_path, res))
  
  averaged_results =
    ddply(fitness_results_df, .(DMS), function(sub_df) {c(median(sub_df[,'SeqSimilarity'], na.rm=T),
                                                          mean(sub_df[,'SeqSimilarity'], na.rm=T),
                                                          median(sub_df[,'BatchCor'], na.rm=T),
                                                          mean(sub_df[,'BatchCor'], na.rm=T),
                                                          mean(sub_df[,'CumCor'], na.rm=T),
                                                          sd(sub_df[,'SeqSimilarity'], na.rm=T),
                                                          sd(sub_df[,'BatchCor'], na.rm=T))
                                                          })
  
  colnames(averaged_results) <- c("DMS", "MedSeqSimilarity", "SeqSimilarity", "MedBatchCor", "BatchCor", "CumCor",
                                  "sdSeqSimilarity", "sdBatchCor")
  all_cor_plots <- list()
  
  for (dms in example_dms) {
    full_values_dms = full_values[full_values$DMS == dms,]
    
    
    g <-
    ggplot(full_values_dms, aes(x=Fitness, y=Predicted_Fitness)) + 
      geom_point(color="navyblue") + 
      theme_light() + 
      ggtitle(sprintf("%s Spearman: %.3f", substr(dms,0,15), cor(full_values_dms$Fitness, full_values_dms$Predicted_Fitness))) +
      theme(title=element_text(size=8))
    
    
    all_cor_plots <- append(all_cor_plots, list(g))
    
  }
  
  all_cor_plots$nrow <- 1
  gexamples_of_cor <- do.call(plot_grid, all_cor_plots)
  
  
  gseq <- 
  ggplot(averaged_results) + 
    geom_point(aes(x=DMS, y=SeqSimilarity)) +
    geom_linerange(aes(x=DMS, ymin=SeqSimilarity - sdSeqSimilarity, ymax = SeqSimilarity + sdSeqSimilarity)) +
    theme_classic() +
    theme(axis.text.x = element_text(size=6,angle = 50, vjust = 1, hjust=1)) +
    ylab("Sequence similarity") + 
    xlab("") + 
    ggtitle(model_name)

  gbcor <- 
    ggplot(averaged_results) + 
    geom_point(aes(x=DMS, y=MedBatchCor)) +
    geom_linerange(aes(x=DMS, ymin=MedBatchCor - sdBatchCor, ymax = MedBatchCor + sdBatchCor)) +
    theme_classic() +
    theme(axis.text.x = element_text(size=6,angle = 50, vjust = 1, hjust=1)) +
    ylab("Spearman R (batches)") + 
    xlab("") + 
    ggtitle(model_name)
   
  gcor <- 
    ggplot(averaged_results) + 
    geom_point(aes(x=DMS, y=CumCor)) +
    theme_classic() +
    theme(axis.text.x = element_text(size=6,angle = 50, vjust = 1, hjust=1)) +
    ylab("Spearman R") + 
    xlab("") +
    ggtitle(model_name)
  
  
  averaged_results$model <- model_name
  
  all_average_results <- rbind(all_average_results,
                               averaged_results)
  
  write_path = sprintf("%s/%s", figures_path, model_name)
  dir.create(write_path)
  
  pdf(sprintf("%s/cor_examples.pdf", write_path), 9,3)
  plot(gexamples_of_cor)
  dev.off()
  pdf(sprintf("%s/batch_cor.pdf", write_path), 9,6)
  plot(gbcor)
  dev.off()
  pdf(sprintf("%s/seqsim.pdf", write_path), 9,6)
  plot(gseq)
  dev.off()
  pdf(sprintf("%s/cumcor.pdf", write_path), 9,6)
  plot(gcor)
  dev.off()
}
gcorall <- 
  ggplot(all_average_results) + 
  geom_point(aes(x=DMS, y=CumCor, color=model, group=model)) +
  theme_classic() +
  theme(axis.text.x = element_text(size=6,angle = 50, vjust = 1, hjust=1)) +
  ylab("Spearman R") + 
  xlab("") +
  scale_color_brewer(palette = "Spectral")



gseqall <- 
  ggplot(all_average_results) + 
  geom_point(aes(x=DMS, y=SeqSimilarity, color=model, group=model)) +
  geom_linerange(aes(x=DMS, ymin=SeqSimilarity - sdSeqSimilarity, ymax = SeqSimilarity + sdSeqSimilarity,
                     color=model, group=model)) +
  theme_classic() +
  theme(axis.text.x = element_text(size=6,angle = 50, vjust = 1, hjust=1)) +
  ylab("Sequence similarity") + 
  xlab("") +
  scale_color_brewer(palette = "Spectral")

  write_path = sprintf("%s/cum_cor_all.pdf", figures_path)
  pdf(write_path, 9,6)
  plot(gcorall)
  dev.off()
  
  write_path = sprintf("%s/seq_sim_all.pdf", figures_path)
  pdf(write_path, 9,6)
  plot(gseqall)
  dev.off()
  
  
  all_model_names <- unique(all_average_results$model)
  
  seq_sim_mt = do.call(cbind, lapply(all_model_names, 
                                     function(md) {all_average_results$SeqSimilarity[all_average_results$model == md]}))
  
  
  cum_cor_mt = do.call(cbind, lapply(all_model_names, 
                                     function(md) {all_average_results$CumCor[all_average_results$model == md]}))
  
  
  
  colnames(cum_cor_mt) <- c(all_model_names)
  colnames(seq_sim_mt) <-  c(all_model_names)
  
  phcc <- pheatmap(cor(cum_cor_mt), cluster_rows=F, cluster_cols=F)
  phsim <- pheatmap(cor(seq_sim_mt), cluster_rows=F, cluster_cols=F)

  
  plot_grid(phsim[[4]], phcc[[4]])
  