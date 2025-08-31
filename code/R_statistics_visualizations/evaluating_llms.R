
library(cowplot)
library(ggplot2)


seq_df = read.csv("/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning/data/configuration/fixed_unique_gfp_sequence_dataset_full_seq.csv")




full_only_df = read.csv("/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning/evaluated_models/full_mask_esm/test_results.csv")
both_df = read.csv("/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning/retraining_esm/model_esm2_t33_650M_UR50D_loss_nll_dkl_orpo_max_diff_indices_muts_1_2_3_mask_type_both_lr_0.00001000_wd_0.100_iter_20000_bs_20/model_esm2_t33_650M_UR50D_loss_nll_dkl_orpo_max_diff_indices_muts_1_2_3_mask_type_both_lr_0.00001000_wd_0.100_iter_20000_bs_20/test_results.csv")
partaial_only_df = read.csv("/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning/retraining_esm/model_esm2_t33_650M_UR50D_loss_nll_dkl_orpo_max_diff_indices_muts_1_2_3_mask_type_partial_lr_0.00001000_wd_0.100_iter_20000_bs_20/model_esm2_t33_650M_UR50D_loss_nll_dkl_orpo_max_diff_indices_muts_1_2_3_mask_type_partial_lr_0.00001000_wd_0.100_iter_20000_bs_20/test_results.csv")
colnames(partaial_only_df) <- c("#", "mask", "pfit", "ffit", "lbl", "idx")
colnames(both_df) <- c("#", "mask", "pfit", "ffit", "lbl", "idx")
colnames(full_only_df) <- c("#", "mask", "pfit", "ffit", "lbl", "idx")

full_only_df$nmuts <- seq_df[full_only_df$idx +1,]$num_muts
both_df$nmuts <- seq_df[both_df$idx + 1,]$num_muts
partaial_only_df$nmuts <- seq_df[partaial_only_df$idx + 1,]$num_muts

best_ind_both <- order(both_df$ffit, decreasing=T)[1:100]
best_ind_partial <- order(partaial_only_df$ffit, decreasing=T)[1:100]


plot_list <- list()
partial_plot_list <- list()
full_plot_list <- list()
n_seq = 50
for (nm in sort(unique(full_only_df$nmuts))[2:7]) {
  
  
  both_ind <- which(both_df$nmuts == nm)[order(both_df[both_df$nmuts == nm,]$pfit, decreasing=T)[1:n_seq]]
  partial_ind <- which(partaial_only_df$nmuts == nm)[order(partaial_only_df[partaial_only_df$nmuts == nm,]$pfit, decreasing=T)[1:n_seq]]
  full_ind <- which(full_only_df$nmuts == nm)[order(full_only_df[full_only_df$nmuts == nm,]$ffit, decreasing=T)[1:n_seq]]
  
  
  gfull <- 
    ggplot(full_only_df[full_ind,]) + 
    geom_histogram(aes(x=ffit, fill=as.factor(lbl)), alpha=.5, position='identity') + 
    xlab("Predicted fitness") + 
    ylab("Frequency") +
    ggtitle(sprintf("Top 100 sequences with\n%d mutations", nm)) +
    scale_fill_manual(name="", values = c("red", "gray50"), labels = c("Actives", "Inactives")) +
    theme(title = element_text(size=10))
  
  full_plot_list <- append(full_plot_list, list(gfull))
  
  gboth <- 
  ggplot(both_df[both_ind,]) + 
    geom_histogram(aes(x=pfit, fill=as.factor(lbl)), alpha=.5, position='identity') + 
    xlab("Predicted fitness") + 
    ylab("Frequency") +
    ggtitle(sprintf("Top 100 sequences with\n%d mutations", nm)) +
    scale_fill_manual(name="", values = c("red", "gray50"), labels = c("Actives", "Inactives")) +
    theme(title = element_text(size=10))
  
  plot_list <- append(plot_list, list(gboth))
  
  gpartial <- 
    ggplot(partaial_only_df[partial_ind,]) + 
      geom_histogram(aes(x=pfit, fill=as.factor(lbl)), alpha=.5, position='identity') + 
      xlab("Predicted fitness") + 
      ylab("Frequency") +
      ggtitle(sprintf("Top 100 sequences with\n%d mutations", nm)) +
      scale_fill_manual(name="", values = c("red", "gray50"), labels = c("Actives", "Inactives"))  +
      theme(title = element_text(size=10))
  
  
  partial_plot_list <- append(partial_plot_list, list(gpartial))
}
  

full_plot_list$nrow <- 2
do.call(plot_grid,full_plot_list)

plot_list$nrow <- 2
do.call(plot_grid,plot_list)

partial_plot_list$nrow <- 2
do.call(plot_grid,partial_plot_list)

g_both <- 
  ggplot(both_df[best_ind_both,]) + 
  geom_histogram(aes(x=ffit, fill=as.factor(lbl)), alpha=.5, position='identity') 

g_partial <- 
  ggplot(partaial_only_df[best_ind_partial,]) + 
  geom_histogram(aes(x=ffit, fill=as.factor(lbl)), alpha=.5, position='identity') 
#geom_density(aes(x=pfit, fill=as.factor(lbl)), alpha=.5)


g_density_both <- 
  ggplot(both_df) + geom_density(aes(x=ffit, fill=as.factor(lbl)), alpha=.5)

g_density_partial <- 
  ggplot(partaial_only_df) + geom_density(aes(x=ffit, fill=as.factor(lbl)), alpha=.5)

plot_grid(g_both, g_partial, nrow=1)
best_ind_both <- order(both_df$pfit, decreasing=T)[1:100]
best_ind_partial <- order(partaial_only_df$pfit, decreasing=T)[1:100]

g_both <- 
ggplot(both_df[best_ind_both,]) + 
  geom_histogram(aes(x=pfit, fill=as.factor(lbl)), alpha=.5, position='identity') 

g_partial <- 
  ggplot(partaial_only_df[best_ind_partial,]) + 
  geom_histogram(aes(x=pfit, fill=as.factor(lbl)), alpha=.5, position='identity') 
  #geom_density(aes(x=pfit, fill=as.factor(lbl)), alpha=.5)



g_density_both <- 
  ggplot(both_df) + geom_density(aes(x=pfit, fill=as.factor(lbl)), alpha=.5)

g_density_partial <- 
  ggplot(partaial_only_df) + geom_density(aes(x=pfit, fill=as.factor(lbl)), alpha=.5)

plot_grid(g_both, g_partial, nrow=1)


df <- full_only_df 

hyp_spec = list()
mean_spec = list()
ind = c()
for(m_id in unique(df$mask)) {
 
  print(m_id)
  tdf = df[df$mask == m_id,]
  
  if (sum(tdf$lbl == 0) > 0 && sum(tdf$lbl == 1) > 0) {
    
    
    ma = max(tdf[tdf$lbl == 0,]$pfit)
    mi = max(tdf[tdf$lbl == 1,]$pfit)
    if (ma > mi) {
      print(sprintf("%.3f - %.3f", ma, mi))
      ind <- c(ind, m_id)
    }
    hyp_spec = append(hyp_spec,
                      list(tdf))
  }
}

count_mt = lapply(hyp_spec, function(tdf) {c(sum(tdf$lb == 0), sum(tdf$lb == 1))})
count_mt <- (do.call(rbind, count_mt))

interesting_hypothesis = which(count_mt[,1] > 20 & count_mt[,2] > 20)


for (idx in 1:len(interesting_hypothesis)) {
  
  hyp_idx = interesting_hypothesis[idx]
  hyp = hyp_spec[[hyp_idx]]
  
  act = hyp[hyp$lbl == 0,]$pfit
  inact = hyp[hyp$lbl == 1,]$pfit
  if (idx == 1) {
    plot(x=rep(idx, len(act)),y=act, xlim=c(0, len(interesting_hypothesis)), ylim=c(min(df$pfit), max(df$pfit)),
         col=adjustcolor("red", alpha=.5), pch=19,
         ylab="Predicted fitness", xlab="Hypothesis ID")
    points(x=rep(idx, len(inact)), y=inact, col=adjustcolor("grey80", alpha=.2), pch=19)
  } else {
    points(x=rep(idx, len(act)),y=act, col=adjustcolor("red", alpha=.5), pch=19)
    points(x=rep(idx, len(inact)), y=inact, col=adjustcolor("grey80", alpha=.2), pch=19)
  }
  
}

min_max_norm <- function(vec) {(vec - min(vec)) / (max(vec) - min(vec))}
normd <- lapply(hyp_spec, function(hyp_df) {min_max_norm(hyp_df$pfit)})
lbls <- lapply(hyp_spec, function(hyp_df) {hyp_df$lbl})
nrmdf <- cbind(unlist(normd), unlist(lbls))
nrmdf <- as.data.frame(nrmdf)
colnames(nrmdf) <- c("normed_fitness", "labels")
ggplot(nrmdf)+#[order(nrmdf$normed_fitness, decreasing=T)[1:500],]) + 
  geom_density(aes(x=normed_fitness, fill=as.factor(labels)))#, position="identity", alpha=.5, bins=50)


