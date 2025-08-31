library(cowplot)
library(ggplot2)
library(pheatmap)

root_path = "/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning"
results_path = sprintf("%s/results/", root_path)
base_path = sprintf("%s/data/configuration/", root_path)
figures_path = sprintf("%s/figures/gfp_dataset/esm3", results_path)
dir.create(figures_path)

fitness_df = read.csv(sprintf("%s/gfp_dataset/esm3/predicted_fitness.csv",results_path))
seq_df = read.csv(paste(base_path, "fixed_unique_gfp_sequence_dataset_full_seq.csv", sep=""))

colnames(fitness_df) <- c("#", "1f", "2f", "1f_structure", "2f_structure")


discrimination_df <- c()

filter_by_count_func <- function(seq_df, count_colnames, count_threshold, is_any=F) {
  
  boolean_indices <- lapply(count_colnames, 
                            function(coln) {r <- seq_df[,coln]; 
                            r[is.na(r)] <- 0; 
                            lst <- list(r >= count_threshold,
                                        r < count_threshold)
                            return(lst)})
  
  active_by_count <- boolean_indices[[1]][[1]]
  inactive_by_count <- boolean_indices[[1]][[2]]
  
  for (col_indices in boolean_indices[-1]) {
    if (is_any) {
      active_by_count <- active_by_count | col_indices[[1]]
      inactive_by_count <- inactive_by_count & col_indices[[2]]
    } else {
      active_by_count <- active_by_count & col_indices[[1]]
      inactive_by_count <- inactive_by_count | col_indices[[2]]
    }
  }
  
  return(list(active=active_by_count, inactive=inactive_by_count))
}
configuration = list('any'=list(condition_func=function(seq_df) {active = seq_df[,"is_unsorted"] == "False"; return(active)},
                                count_colnames=c("L_GFP_counts", "H_GFP_counts", "L_AmCyan_counts", "H_AmCyan_counts", "L_any_counts", "H_any_counts"),
                                is_any=T,
                                title="Active vs Inactive sequences",
                                color="red"),
                     'gfp'=list(condition_func=function(seq_df) {active = (rowSums(seq_df[,c("L_GFP","H_GFP")]) >= 1); return(active)},
                                count_colnames=c("L_GFP_counts", "H_GFP_counts"),
                                is_any=F,
                                title="GFP vs non-GFP sequences",
                                color="green"),
                     'amcyan'=list(condition_func=function(seq_df) {active = (rowSums(seq_df[,c("L_AmCyan","H_AmCyan")]) >= 1); return(active)},
                                   count_colnames=c("L_AmCyan_counts", "H_AmCyan_counts"),
                                   is_any=F,
                                   title="AmCyan vs\n Non-AmCyan sequences",
                                   color="cyan"))
                     # 'gfpamcyan'=list(condition_func=function(seq_df) {active = (rowSums(seq_df[,c("L_any","H_any")]) >= 1); return(active)},
                     #                  count_colnames=c("L_any_counts", "H_any_counts"),
                     #                  is_any=F,
                     #                  title="GFP+AmCyan vs\n Non-(Gfp+AmCyan) sequences",
                     #                  color="yellow"))




balance = FALSE

for (col_to_use in 2:5) {

for (conf_name in names(configuration)) {
  
  conf = configuration[[conf_name]]
  

  
  separation <- c()
  condition_active = conf$condition_func(seq_df)
  tested_thresholds = c(seq(0,5, by=1), 10, 20, 50, 100)
  
  plot_list <- list()
  
  for (count_t in tested_thresholds) {
    
    count_filtered_indices = filter_by_count_func(seq_df, 
                                                  count_colnames = conf$count_colnames, 
                                                  count_threshold = count_t, 
                                                  is_any=conf$is_any)
    
    active_sequences <- condition_active & conf$condition_func(seq_df) & count_filtered_indices$active
    inactive_sequences <- !active_sequences | count_filtered_indices$inactive
    
    median_active = median(fitness_df[active_sequences,col_to_use])
    median_inactive = median(fitness_df[inactive_sequences,col_to_use])
    separation <- c(separation, median_active - median_inactive)
    
    
    if (count_t %in% c(5, 20)) {
      stats_str = sprintf("[Count threshold %d, A:%d, I:%d]",
                          count_t, 
                          sum(active_sequences),
                          sum(inactive_sequences))
      if (count_t == 1) {
        mt = sprintf("%s\n%s", 
                     conf$title, 
                     stats_str)
      } else {
        mt = stats_str
      }
      
      tmp_df = fitness_df
      tmp_df$active_inactive <- rep("", times=nrow(fitness_df))
      tmp_df$active_inactive[active_sequences] = 1
      tmp_df$active_inactive[inactive_sequences] = 0
      colnames(tmp_df)[col_to_use] <- "fitness"
      
      if (balance) {
        tmp_df <- rbind(tmp_df[active_sequences,],
                        tmp_df[sample(which(inactive_sequences), len(which(active_sequences))),])
      }
      # h1 = hist(fitness_df[active_sequences,col_to_use], prob=T, breaks=50, plot=F)
      # h2 = hist(fitness_df[inactive_sequences,col_to_use], prob=T, add=T, breaks=50, plot=F)
      # max_density = max(max(h2$density), max(h1$density))
      # hist(fitness_df[inactive_sequences,col_to_use], prob=T, ylim=c(0,max_density), xlab='ESM zero-shot fitness score', main=mt, breaks=50)
      # hist(fitness_df[active_sequences,col_to_use], prob=T, add=T, col=adjustcolor(conf$color, alpha=.2), breaks=50)
      # abline(v=mean_inactive, lty=2, col="black", lwd=1.5)
      # abline(v=mean_active, lty=2, col=conf$color, lwd=1.5)
      
      h_active = hist(tmp_df[tmp_df$active_inactive == "1",col_to_use], breaks=50, plot=F, prob=T)
      h_inactive = hist(tmp_df[tmp_df$active_inactive == "0",col_to_use], breaks=50, plot=F, prob=T)
      
      hinactive_breaks = sapply(1:(len(h_inactive$breaks) - 1), 
                                function(idx) {mean(c(h_inactive$breaks[idx], h_inactive$breaks[idx + 1]))})
      
      hactive_breaks = sapply(1:(len(h_active$breaks) - 1), 
                              function(idx) {mean(c(h_active$breaks[idx], h_active$breaks[idx + 1]))})
      
      hinactive_density = h_inactive$density
      hactive_density = h_active$density
      
      # hist_df = data.frame(breaks=c(hactive_breaks, hinactive_breaks),
      #                      density=c(hactive_density, hinactive_density))
                           #labels=c(rep(c("1", "0"), times=c(len(hactive_density), len(hinactive_density)))))
      
      active_hist_df = data.frame(breaks=c(hactive_breaks),
                                  density=c(hactive_density))
      
      inactive_hist_df = data.frame(breaks=c(hinactive_breaks),
                                    density=c(hinactive_density))
      g <- 
      ggplot(tmp_df) + 
        geom_bar(data=active_hist_df, 
                 aes(x=breaks, y=density),
                 stat="identity", 
                 alpha=.15,
                 color="gray50",
                 fill=conf$color,
                 linewidth=.25,
                 position="dodge") + 
        geom_bar(data=inactive_hist_df, 
                 aes(x=breaks, y=density), 
                 stat="identity", 
                 alpha=.15,
                 color="gray50",
                 fill="black",
                 linewidth=.25,
                 position="dodge") +
        geom_density(aes(x=fitness, 
                         group=active_inactive, 
                         fill=active_inactive), color="black", alpha=.2) + 
        theme_classic() + scale_y_continuous(expand=c(0,0)) + xlab("ESM zero-shot fitness score") + ylab("Density") + theme(text=element_text(color="black")) + 
        ggtitle(sprintf("%s %s", conf_name, stats_str)) + 
        theme(legend.position = "none") + 
        scale_fill_manual(breaks=c('1', '0'), values=c(conf$color, "gray50")) +
        geom_vline(xintercept=median_active, col=conf$color, linetype="dashed", linewidth=1)+#, linewidth=1.15) + 
        geom_vline(xintercept=median_inactive, col="black", linetype="dashed", linewidth=1) #, linewidth=1.15) + 

      
      plot_list <- append(plot_list, list(g))
      
    }
  }
  
  
  
  threshold_df = data.frame(`x`=tested_thresholds,
                            `y`=separation)
  
  sep_df <- data.frame(separation=separation,
                       thresholds=tested_thresholds)
  
  sep_df$model = paste("esm3_", colnames(fitness_df)[col_to_use], sep="")
  sep_df$conf = conf_name
  
  discrimination_df <- rbind(discrimination_df,
                             sep_df)
  
  
  g <- 
  ggplot(threshold_df, aes(x=x,y=y)) +
    geom_point() +
    theme_classic() +
    xlab("Count threshold") +
    ylab("   Median(fitness | active)\n - Median(fitness | inactive)") +
    theme(text=element_text(color="black"))
  
  
  plot_list = append(plot_list, list(g))
  plot_list$nrow = 1
  
  #plot(x=tested_thresholds, y=separation, xlab="Count threshold", ylab="Mean(fitness | active) - Mean(fitness | inactive)")
  
  figure_filename = sprintf("%s/%sesm3_%s_%s_with_hists.pdf", 
                            figures_path,
                            ifelse(balance, "balanced_", ""),
                            conf_name, 
                            colnames(fitness_df)[col_to_use])
  pdf(figure_filename, 9,3)
  plt <- do.call(plot_grid, plot_list)
  plot(plt)
  dev.off()
  
}
}


cmt <- pheatmap(cor(fitness_df[,-1]), 
                cluster_rows=F,
                cluster_cols=F)

figure_filename = sprintf("%s/esm3_cor_mat.pdf", figures_path)
pdf(figure_filename, 4,4)
plot(cmt[[4]])
dev.off()


esm2_models = c("esm_msa1b_t12_100M_UR50S",
                "esm2_t33_650M_UR50D",
                "esm1b_t33_650M_UR50S",
                "esm1_t34_670M_UR100",
                "esm1v_t33_650M_UR90S_1",                       
                "esm2_t36_3B_UR50D")

figures_path = sprintf("%s/figures/gfp_dataset/esm2", results_path)
dir.create(figures_path)

all_fitness_df <- c()
for (esm_2_model_name in esm2_models) {
  
  fitness_df = read.csv(sprintf("%s/gfp_dataset/esm2/%s_predicted_fitness.csv",results_path, esm_2_model_name))
  
  all_fitness_df <- cbind(all_fitness_df,
                          fitness_df[,2])
  col_to_use <- 2 
  for (conf_name in names(configuration)) {
    
    conf = configuration[[conf_name]]
    
    
    
    separation <- c()
    condition_active = conf$condition_func(seq_df)
    tested_thresholds = c(seq(0,5, by=1), 10, 20, 50, 100)
    
    plot_list <- list()
    
    for (count_t in tested_thresholds) {
      
      count_filtered_indices = filter_by_count_func(seq_df, 
                                                    count_colnames = conf$count_colnames, 
                                                    count_threshold = count_t, 
                                                    is_any=conf$is_any)
      
      active_sequences <- condition_active & conf$condition_func(seq_df) & count_filtered_indices$active
      inactive_sequences <- !active_sequences | count_filtered_indices$inactive
      
      median_active = median(fitness_df[active_sequences,col_to_use])
      median_inactive = median(fitness_df[inactive_sequences,col_to_use])
      separation <- c(separation, median_active - median_inactive)
      
      
      if (count_t %in% c(5, 20)) {
        stats_str = sprintf("[Count threshold %d, A:%d, I:%d]",
                            count_t, 
                            sum(active_sequences),
                            sum(inactive_sequences))
        if (count_t == 1) {
          mt = sprintf("%s\n%s", 
                       conf$title, 
                       stats_str)
        } else {
          mt = stats_str
        }
        
        tmp_df = fitness_df
        tmp_df$active_inactive <- rep("", times=nrow(fitness_df))
        tmp_df$active_inactive[active_sequences] = 1
        tmp_df$active_inactive[inactive_sequences] = 0
        colnames(tmp_df)[2] <- "fitness"
        tmp_df <- cbind(tmp_df, seq_df$num_of_muts)
        colnames(tmp_df)[4] <- "num_of_muts"
        
        if (balance) {
          tmp_df <- rbind(tmp_df[active_sequences,],
                          tmp_df[sample(which(inactive_sequences), len(which(active_sequences))),])
        }
        # h1 = hist(fitness_df[active_sequences,col_to_use], prob=T, breaks=50, plot=F)
        # h2 = hist(fitness_df[inactive_sequences,col_to_use], prob=T, add=T, breaks=50, plot=F)
        # max_density = max(max(h2$density), max(h1$density))
        # hist(fitness_df[inactive_sequences,col_to_use], prob=T, ylim=c(0,max_density), xlab='ESM zero-shot fitness score', main=mt, breaks=50)
        # hist(fitness_df[active_sequences,col_to_use], prob=T, add=T, col=adjustcolor(conf$color, alpha=.2), breaks=50)
        # abline(v=mean_inactive, lty=2, col="black", lwd=1.5)
        # abline(v=mean_active, lty=2, col=conf$color, lwd=1.5)
        
        h_active = hist(tmp_df[tmp_df$active_inactive == "1",col_to_use], breaks=50, plot=F, prob=T)
        h_inactive = hist(tmp_df[tmp_df$active_inactive == "0",col_to_use], breaks=50, plot=F, prob=T)
        
        hinactive_breaks = sapply(1:(len(h_inactive$breaks) - 1), 
                                  function(idx) {mean(c(h_inactive$breaks[idx], h_inactive$breaks[idx + 1]))})
        
        hactive_breaks = sapply(1:(len(h_active$breaks) - 1), 
                                function(idx) {mean(c(h_active$breaks[idx], h_active$breaks[idx + 1]))})
        
        hinactive_density = h_inactive$density
        hactive_density = h_active$density
        
        # hist_df = data.frame(breaks=c(hactive_breaks, hinactive_breaks),
        #                      density=c(hactive_density, hinactive_density))
        #labels=c(rep(c("1", "0"), times=c(len(hactive_density), len(hinactive_density)))))
        
        active_hist_df = data.frame(breaks=c(hactive_breaks),
                                    density=c(hactive_density))
        
        inactive_hist_df = data.frame(breaks=c(hinactive_breaks),
                                      density=c(hinactive_density))
        
        
        # tmp_df[tmp_df$num_of_muts <= max(tmp_df[tmp_df$active_inactive == '1',"num_of_muts"]),]
        #g <- 
          ggplot(tmp_df) + 
          # geom_bar(data=active_hist_df, 
          #          aes(x=breaks, y=density),
          #          stat="identity", 
          #          alpha=.15,
          #          color="gray50",
          #          fill=conf$color,
          #          linewidth=.25,
          #          position="dodge") + 
          # geom_bar(data=inactive_hist_df, 
          #          aes(x=breaks, y=density), 
          #          stat="identity", 
          #          alpha=.15,
          #          color="gray50",
          #          fill="black",
          #          linewidth=.25,
          #          position="dodge") +
          geom_violin(aes(y=fitness,
                          x=active_inactive,
                           group=active_inactive, 
                           fill=active_inactive), color="black", alpha=.2) + 
          theme_classic() + scale_y_continuous(expand=c(0,0)) + xlab("ESM zero-shot fitness score") + ylab("Density") + theme(text=element_text(color="black")) + 
          ggtitle(sprintf("%s %s", conf_name, stats_str)) + 
          theme(legend.position = "none") + 
          scale_fill_manual(breaks=c('1', '0'), values=c(conf$color, "gray50")) +
          geom_vline(xintercept=median_active, col=conf$color, linetype="dashed", linewidth=1)+#, linewidth=1.15) + 
          geom_vline(xintercept=median_inactive, col="black", linetype="dashed", linewidth=1) #, linewidth=1.15) + 
        
        
          plot_list <- append(plot_list, list(g))
        
      }
    }
    
    threshold_df = data.frame(`x`=tested_thresholds,
                              `y`=separation)
    
    
    sep_df <- data.frame(separation=separation,
                         thresholds=tested_thresholds)
    
    sep_df$model = esm_2_model_name
    sep_df$conf = conf_name
    
    discrimination_df <- rbind(discrimination_df,
                               sep_df)
    
    
    g <- 
    g <- 
      ggplot(threshold_df, aes(x=x,y=y)) +
      geom_point() +
      theme_classic() +
      xlab("Count threshold") +
      ylab("   Median(fitness | active)\n - Median(fitness | inactive)") +
      theme(text=element_text(color="black"))
    
    
    plot_list = append(plot_list, list(g))
    plot_list$nrow = 1
    
    #plot(x=tested_thresholds, y=separation, xlab="Count threshold", ylab="Mean(fitness | active) - Mean(fitness | inactive)")
    
    figure_filename = sprintf("%s/%sesm2_%s_%s_with_hists.pdf", 
                              figures_path,
                              ifelse(balance, "balanced_", ""),
                              conf_name, 
                              esm_2_model_name)
    pdf(figure_filename, 9,3)
    plt <- do.call(plot_grid, plot_list)
    plot(plt)
    dev.off()
    
  }
}

esm3_fitness_df = read.csv(sprintf("%s/gfp_dataset/esm3/predicted_fitness.csv",results_path))


all_fitness_df <- cbind(all_fitness_df, esm3_fitness_df[,-c(1,2)])

colnames(all_fitness_df) <- c(esm2_models, "esm3 1-forward")
colnames(esm3_fitness_df) <- c("#", paste("esm3 ", c("1-forward", "2-forward", "1-forward + fixed backbone", "2-forward + fixed backbone")))

pheatmap(cor(all_fitness_df), cluster_rows=F, cluster_cols=F)
pheatmap(cor(esm3_fitness_df[,-1]), cluster_rows=F, cluster_cols=F)
all_with_esm3 <- rbind(all_fitness_df, esm3_fitness_df)

pheatmap(cor(all_fitness_df), cluster_rows=F, cluster_cols=F)
colnames(all_with_esm3) <- c()

grid <- c()
sq <- seq(0, 2, length.out=200)
for (i in 1:200) {for (j in 1:200) { grid <- rbind(grid, c(sq[i], sq[j]))}}


mu1 <- c(2, 2)
sigma1 <- matrix(c(1, 0.5, 0.5, 1), 2)

z1 <- dmvnorm(grid, mean = mu1, sigma = sigma1)
z2 <- rowSums(grid)


gridf <- as.data.frame(cbind(grid, z1, z2))
colnames(gridf) <- c("x", "y", "z1", "z2")
gridf$z3 <- gridf$x

gridf$z1 <- (gridf$z1 - min(gridf$z1)) / (max(gridf$z1) - min(gridf$z1))
gridf$z2 <- (gridf$z2 - min(gridf$z2)) / (max(gridf$z2) - min(gridf$z2))
gridf$z3 <- (gridf$z3 - min(gridf$z3)) / (max(gridf$z3) - min(gridf$z3))


g1 <- 
  ggplot(gridf, aes(x = x, y = y, fill = z1)) +
    geom_raster(interpolate = TRUE) +
    theme_classic()+
    scale_fill_viridis_c() +
    theme(
      #axis.text = element_blank(),
      #axis.line = element_blank(),
      axis.ticks = element_blank())+
      #legend.position= "none") +
    ylab("Mutation complexity (AU)") +
    xlab("Sample size (AU)") + 
    labs(fill = "Classification\nAccuracy (%)") +
  ggtitle("Epsitatic protein landscape")
    

g2 <- 
  ggplot(gridf, aes(x = x, y = y, fill = z2)) +
    geom_raster(interpolate = TRUE) +
    theme_classic()+
    scale_fill_viridis_c() +
    theme(
      #axis.text = element_blank(),
      #axis.line = element_blank(),
      axis.ticks = element_blank())+
      #legend.position= "none") +
    ylab("Mutation complexity (AU)") +
    xlab("Sample size (AU)") +
    labs(fill = "Classification\nAccuracy (%)") +
  ggtitle("Non-Trivial linear protein landscape")


g3 <- 
  ggplot(gridf, aes(x = x, y = y, fill = z3)) +
  geom_raster(interpolate = TRUE) +
  theme_classic()+
  scale_fill_viridis_c() +
  theme(
    #axis.text = element_blank(),
    #axis.line = element_blank(),
    axis.ticks = element_blank())+
  #legend.position= "none") +
  ylab("Mutation complexity (AU)") +
  xlab("Sample size (AU)") +
  labs(fill = "Classification\nAccuracy (%)") +
  ggtitle("Completely additive mutation landscape")


plot_grid(g1, g2, g3, nrow=1)
gdiscrim <- 
ggplot(discrimination_df[discrimination_df$thresholds %in% c(10,100) &
                         !discrimination_df$model %in% c("esm3_1f_structure", "esm3_2f_structure", "esm3_2f"),], 
                        aes(x=model, y=separation, group=interaction(conf, thresholds))) +
  geom_point(aes(group=interaction(conf, thresholds), color=interaction(conf, thresholds))) +
  geom_line(aes(group=interaction(conf, thresholds), color=interaction(conf, thresholds))) +
  ylab("Sepearation index") +
  xlab("Model") +
  theme_classic() +
  theme(axis.text.x = element_text(size=10,angle = 50, vjust = 1, hjust=1),
        legend.title=element_blank()) +
  scale_color_brewer(palette = "Spectral")


gdiscrim_esm3 <- 
  ggplot(discrimination_df[discrimination_df$thresholds %in% c(10,100) &
                          discrimination_df$model %in% c("esm3_1f", "esm3_1f_structure", "esm3_2f_structure", "esm3_2f"),], 
         aes(x=model, y=separation, group=interaction(conf, thresholds))) +
  geom_point(aes(group=interaction(conf, thresholds), color=interaction(conf, thresholds))) +
  geom_line(aes(group=interaction(conf, thresholds), color=interaction(conf, thresholds))) +
  ylab("Sepearation index") +
  xlab("Model") +
  theme_classic() +
  theme(axis.text.x = element_text(size=10,angle = 50, vjust = 1, hjust=1),
        legend.title=element_blank()) +
    scale_color_brewer(palette = "Spectral")


figures_path = sprintf("%s/figures/gfp_dataset/", results_path)
write_path = sprintf("%s/discrimination_no_esm3_all.pdf", figures_path)
pdf(write_path, 9,6)
plot(gdiscrim)
dev.off()

write_path = sprintf("%s/discrimination_esm3_all.pdf", figures_path)
pdf(write_path, 9,6)
plot(gdiscrim_esm3)
dev.off()

vocab = c("L", "A", "G", "V", "S", 
          "E", "R", "T", "I", "D",
          "P", "K", "Q", "N", "F",
          "Y", "M", "H", "W", "C", 
          "X", "B", "U", "Z", "O")

designed_muts = c('L42', 'L44', 'V61', 'T65', 'V68', 'Q69', 'S72', 'Q94', 'T108', 'V112', 'N121', 'Y145', 'H148', 'V150', 'T167', 'H181', 'N185', 'T203', 'S205', 'L220', 'E222', 'V224')

designed_pssm_path = "/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning/results/gfp_dataset/designed_positions_pssm.csv"
full_pssm_path = "/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning/results/gfp_dataset/full_pssm.csv"

designed_pssm = read.csv(designed_pssm_path)
full_pssm = read.csv(full_pssm_path)

colnames(full_pssm) <- c("IDX", vocab)
colnames(designed_pssm) <- c("IDX", vocab)
rownames(designed_pssm) <- designed_muts


