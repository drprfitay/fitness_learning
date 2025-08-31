
library(cowplot)
library(ggplot2)


seq_df = read.csv("/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning/data/configuration/fixed_unique_gfp_sequence_dataset_full_seq.csv")
esm_df = read.csv("/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning/evaluated_models/full_mask_esm/test_results.csv")
epinnet_df = read.csv("/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning/evaluated_models/epinnet_1_3/eval.csv")
colnames(esm_df) <- c("#", "mask", "pfit", "ffit", "lbl", "idx")
colnames(epinnet_df) <- c("fit", "lbl", "pred_lbl", "idx")
esm_df$nmuts <- seq_df[esm_df$idx +1,]$num_muts
epinnet_df$nmuts <- seq_df[epinnet_df$idx +1,]$num_muts


epinnet_df <- b_epinnet_df
esm_df <- b_esm_df
# epinnet_df <- epinnet_df[epinnet_df$nmuts == 4,]
# esm_df <- esm_df[esm_df$nmuts == 4,]

for (K in c(50, 100, 500, 1000, 5000)) {
top_K_epinnet <- order(epinnet_df$fit, decreasing = T)[1:K] 
top_K_esm <- order(esm_df$ffit, decreasing = T)[1:K]

  df = data.frame()
  nmuts_df = data.frame()
  for (i in 1:(K / 5)) {
    
    counts_esm = esm_df[top_K_esm[1:(i*5)],]$lbl
    counts_epinnet = epinnet_df[top_K_epinnet[1:(i*5)],]$lbl
    
    nmuts_esm = esm_df[top_K_esm[1:(i*5)],]$nmuts
    nmuts_epinnet = epinnet_df[top_K_epinnet[1:(i*5)],]$nmuts
    
    prob_nmut_esm = table(nmuts_esm) / len(nmuts_esm)
    prob_nmut_epinnet = table(nmuts_epinnet) / len(nmuts_epinnet)
    
    prob_esm = sum(counts_esm == 0) / (i * 5)
    prob_epinnet = sum(counts_epinnet == 0) / (i * 5)
    
    count_df <- 
    data.frame(x=(i * 5 / K),
               prob_esm=prob_esm,
               prob_epinnet=prob_epinnet,
               i=i)
    
    df <- rbind(df, count_df)
    
    # nmuts_df = 
    #   rbind(cbind(as.matrix(prob_nmut_esm), 
    #               as.matrix(as.integer((names(prob_nmut_esm))))),
    #         cbind(as.matrix(prob_nmut_epinnet), 
    #               as.matrix(as.integer((names(prob_nmut_epinnet))))))
    # 
    # nmuts_df <- as.data.frame(nmuts_df)
    # nmuts_df$x = i * 5
    # nmuts_df$group = 
  }
  
  esm_portion_df = df[,c("x", "prob_esm", "i")]
  esm_portion_df$group = "Active sequences"
  wrong_esm_portion_df = esm_portion_df
  wrong_esm_portion_df$prob_esm = 1-esm_portion_df$prob_esm
  wrong_esm_portion_df$group = "Inactive sequences"
  esm_portion_df <- rbind(esm_portion_df, wrong_esm_portion_df)
  
  g_esm <- 
    ggplot(esm_portion_df) + 
    geom_bar(aes(x=x * K, y=prob_esm, fill=group, color=group), 
             stat="identity") + ylab("Cumulative fraction") + 
    xlab("Number of sequences") +
    theme_light() + 
    ggtitle(sprintf("ESM top %d", K))
  
  epinnet_portion_df = df[,c("x", "prob_epinnet", "i")]
  epinnet_portion_df$group = "Active sequences"
  wrong_epinnet_portion_df = epinnet_portion_df
  wrong_epinnet_portion_df$prob_epinnet = 1-epinnet_portion_df$prob_epinnet
  wrong_epinnet_portion_df$group = "Inactive sequences"
  epinnet_portion_df <- rbind(epinnet_portion_df, wrong_epinnet_portion_df)

  g_epinnet <- 
    ggplot(epinnet_portion_df) + 
    geom_bar(aes(x=x * K, y=prob_epinnet, fill=group, color=group), 
             stat="identity") + ylab("Cumulative fraction") + 
    xlab("Number of sequences") +
    theme_light() + 
    ggtitle(sprintf("Epinnet top %d", K))
  
  plot_grid(g_esm, g_epinnet, nrow = 1)
}

 
ggplot(esm_df[top_K_esm,]) + 
  geom_histogram(aes(x=ffit, fill=as.factor(lbl)), alpha=.5, position='identity') + 
  xlab("Predicted fitness") + 
  ylab("Frequency") +
  ggtitle(sprintf("Top 100 sequences with\n%d mutations", nm)) +
  scale_fill_manual(name="", values = c("red", "gray50"), labels = c("Actives", "Inactives")) +
  theme(title = element_text(size=10))
  

ggplot(epinnet_df[top_K_epinnet,]) + 
  geom_histogram(aes(x=fit, fill=as.factor(lbl)), alpha=.5, position='identity') + 
  xlab("Predicted fitness") + 
  ylab("Frequency") +
  ggtitle(sprintf("Top 100 sequences with\n%d mutations", nm)) +
  scale_fill_manual(name="", values = c("red", "gray50"), labels = c("Actives", "Inactives")) +
  theme(title = element_text(size=10))

