


base_path = "/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning/data/configuration/"
fitness_df = read.csv(paste(base_path, "esm_fitness_df.csv", sep=""))
seq_df = read.csv(paste(base_path, "fixed_unique_gfp_sequence_dataset.csv", sep=""))






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
                                color="cyan"),
                    'gfpamcyan'=list(condition_func=function(seq_df) {active = (rowSums(seq_df[,c("L_any","H_any")]) >= 1); return(active)},
                                  count_colnames=c("L_any_counts", "H_any_counts"),
                                  is_any=F,
                                  title="GFP+AmCyan vs\n Non-(Gfp+AmCyan) sequences",
                                  color="yellow"))



for (conf in configuration) {
  
  
  dev.off()
  par(mfrow=c(1,5))
  
  separation <- c()
  condition_active = conf$condition_func(seq_df)
  tested_thresholds = c(seq(0,5,by=1), 10, 20, 50, 100)
  for (count_t in tested_thresholds) {
    
    count_filtered_indices = filter_by_count_func(seq_df, 
                                                  count_colnames = conf$count_colnames, 
                                                  count_threshold = count_t, 
                                                  is_any=conf$is_any)
    
    active_sequences <- condition_active & conf$condition_func(seq_df) & count_filtered_indices$active
    inactive_sequences <- !active_sequences | count_filtered_indices$inactive
    
    mean_active = mean(fitness_df[active_sequences,2])
    mean_inactive = mean(fitness_df[inactive_sequences,2])
    separation <- c(separation, mean_active - mean_inactive)
  
    
    if (count_t %in% c(1, 5, 10, 50)) {
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
        h1 = hist(fitness_df[active_sequences,2], prob=T, breaks=50, plot=F)
        h2 = hist(fitness_df[inactive_sequences,2], prob=T, add=T, breaks=50, plot=F)
        max_density = max(max(h2$density), max(h1$density))
        hist(fitness_df[inactive_sequences,2], prob=T, ylim=c(0,max_density), xlab='ESM zero-shot fitness score', main=mt, breaks=50)
        hist(fitness_df[active_sequences,2], prob=T, add=T, col=adjustcolor(conf$color, alpha=.2), breaks=50)
        abline(v=mean_inactive, lty=2, col="black", lwd=1.5)
        abline(v=mean_active, lty=2, col=conf$color, lwd=1.5)
    }
  }
  
  plot(x=tested_thresholds, y=separation, xlab="Count threshold", ylab="Mean(fitness | active) - Mean(fitness | inactive)")
}


active_indices = which()
inactive_indices = which(seq_df[,"is_unsorted"] == "True")

hist(fitness_df[inactive_indices,2], prob=T, ylim=c(0,2.1), xlab='ESM zero-shot fitness score', main="Active vs Inactive sequences", breaks=50)
hist(fitness_df[active_indices,2], prob=T, add=T, col=adjustcolor("red", alpha=.2), breaks=50)
abline(v=mean(fitness_df[inactive_indices,2]), lty=2, col="black", lwd=1.5)
abline(v=mean(fitness_df[active_indices,2]), lty=2, col="red", lwd=1.5)


GFP_Counts = seq_df[,c("L_GFP_counts","H_GFP_counts")]
GFP_Counts = rowSums(apply(GFP_Counts, 2, function(r) {r[is.na(r)] <- 0; return(r)}))
count_threshold = 5
non_zero_GFP_counts = GFP_Counts > count_threshold 
zero_gfp_counts = GFP_Counts <= count_threshold 
gfp_indices = (rowSums(seq_df[,c("L_GFP","H_GFP")]) >= 1) & non_zero_GFP_counts
non_gfp_indices = (rowSums(seq_df[,c("L_GFP","H_GFP")]) < 1) | zero_gfp_counts



xamcyan_indices = rowSums(seq_df[,c("L_AmCyan","H_AmCyan")]) >= 1
non_amcyan_indices = rowSums(seq_df[,c("L_AmCyan","H_AmCyan")]) < 1

hist(fitness_df[non_amcyan_indices,2], prob=T, ylim=c(0,2.1), xlab='ESM zero-shot fitness score', main="AmCyan vs\n Non-AmCyan sequences", breaks=300, border = NA)
hist(fitness_df[amcyan_indices,2], prob=T, add=T, col=adjustcolor("cyan", alpha=.2), breaks=300, border = NA)
abline(v=mean(fitness_df[non_amcyan_indices,2]), lty=2, col="black", lwd=1.5)
abline(v=mean(fitness_df[amcyan_indices,2]), lty=2, col="cyan", lwd=1.5)

any_indices = rowSums(seq_df[,c("L_any", "H_any")]) >= 1
non_any_indices = rowSums(seq_df[,c("L_any", "H_any")]) < 1

hist(fitness_df[non_any_indices,2], prob=T, ylim=c(0,2.1), xlab='ESM zero-shot fitness score', main="GFP+AmCyan vs\n Non-(Gfp+AmCyan) sequences", breaks=300, border = NA)
hist(fitness_df[any_indices,2], prob=T, add=T, col=adjustcolor("yellow", alpha=.2), breaks=300, border = NA)
abline(v=mean(fitness_df[non_any_indices,2]), lty=2, col="black", lwd=1.5)
abline(v=mean(fitness_df[any_indices,2]), lty=2, col="yellow", lwd=1.5)
