
library(ggplot2)
base_path = "/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning/data/"
big_df_path = "configuration/gfp_sequence_dataset.csv"
train_df_path = "datasets/gfp_train_1/sequences.csv"


all_seq_df = read.csv(sprintf("%s/%s", base_path, big_df_path))
train_df = read.csv(sprintf("%s/%s", base_path, train_df_path))
train_ind = all_seq_df[,"sequence"] %in% train_df[,3] 


all_train_df = all_seq_df[train_ind,]
all_test_df = all_seq_df[!train_ind,]

SAMPLE_SIZE = 100000
get_unique_df <- function(df, uid_seq_column) {
  
  seq_vec <- c(df[,uid_seq_column])
  unique_and_tabled <- table(seq_vec)
  
  one_timers = unique_and_tabled <= 1
  multiple_timers = !one_timers
  multiple_occuring_seq <- names(unique_and_tabled[multiple_timers])
  mul_df <- df[seq_vec %in% multiple_occuring_seq,] # Search in a smaller df every_time
  one_df <- df[seq_vec %in% names(unique_and_tabled[one_timers]),]
  
  multiple_occuring_df_split = lapply(multiple_occuring_seq, function(seq) {mul_df[mul_df[,uid_seq_column] %in% seq,]})
  # sanity lapply(1:len(multiple_occuring_df_split), function(dfidx) {sapply(1:ncol(mul_df), function(idx) {len(unique(multiple_occuring_df_split[[dfidx]][,idx]))})})
  
  unique_df = do.call(rbind, lapply(multiple_occuring_df_split, function(df) {df[1,]}))
  
  final_unique_df <- rbind(unique_df, one_df)
  #assert(len(unique_and_tabled) == nrow(final_unique_df))
  print(len(unique_and_tabled) == nrow(final_unique_df))
  
  rownames(final_unique_df) <- final_unique_df[,uid_seq_column]
  
  
  
  
  return(final_unique_df[order(rownames(final_unique_df)),])
}

uid_seq_column = "sequence"

unique_train_df <- get_unique_sequences(all_train_df, uid_seq_column)
unique_test_df <- get_unique_sequences(all_test_df, uid_seq_column)

unique_df_all <- rbind(unique_test_df, unique_train_df)
unique_df_all <- unique_df_all[order(rownames(unique_df_all)),]
len(unique(unique_df_all[,"sequence"])) == nrow(unique_df_all)
unique_df_all[,"idx"] <- 1:nrow(unique_df_all)

write.csv(unique_df_all, sprintf("%s/%s", base_path, "configuration/fixed_unique_gfp_sequence_dataset.csv"))

random_sample_ind <- sample(1:nrow(unique_test_df), SAMPLE_SIZE)
sampled_df <- unique_test_df[sort(random_sample_ind),]


train_df <- unique_df_all[unique_df_all[,"sequence"] %in% unique_train_df[,"sequence"], c("idx", "sequence")]
test_df <- unique_df_all[random_sample_ind, c("idx", "sequence")]


write.csv(train_df, sprintf("%s/%s", base_path, "configuration/random_100k_train.csv"))
write.csv(test_df, sprintf("%s/%s", base_path, "configuration/random_100k_test.csv"))






gmut_dist <- 
ggplot(sampled_df, aes(x=num_of_muts, y=..density..)) + 
  geom_histogram(binwidth = 1) + 
  geom_histogram(data=all_train_df, binwidth = 1, fill="red", alpha=.2) + 
  theme_classic() +
  xlab("Number of mutations") +
  ylab("Density")



positions = colnames(unique_df_all)[40:ncol(unique_df_all)]


one_hot_list <- lapply(positions, function(p){
  tmp_df <- sampled_df[,p]
  m <- model.matrix(~0+tmp_df)
  colnames(m) <- paste(sort(unique(tmp_df)), p, sep="_")
  return(m)})

one_hot_encoding_test_df <- do.call(cbind, one_hot_list)



one_hot_list <- lapply(positions, function(p){
  print(p)
  tmp_df <- unique_train_df[,p]
  unique_vars <- unique(tmp_df)
  
  if (length(unique_vars) <= 1) {
    ret_df <- data.frame(rep(1,nrow(unique_train_df)))
    colnames(ret_df) <- paste(unique_vars[1], p, sep="_")
    return(ret_df)
  }
  
  m <- model.matrix(~0+tmp_df)
  colnames(m) <- paste(sort(unique_vars), p, sep="_")
  return(m)})

one_hot_encoding_train_df <- do.call(cbind, one_hot_list)


non_variability_positions_train <- colnames(one_hot_encoding_test_df)[!colnames(one_hot_encoding_test_df) %in% colnames(one_hot_encoding_train_df)]

zero_pad_mat <-  matrix(0, nrow=nrow(one_hot_encoding_train_df), ncol=len(non_variability_positions_train))
colnames(zero_pad_mat) <- non_variability_positions_train

one_hot_encoding_train_df <- cbind(one_hot_encoding_train_df, zero_pad_mat)
one_hot_encoding_train_df <- one_hot_encoding_train_df[,colnames(one_hot_encoding_test_df)]
colnames(one_hot_encoding_train_df) == colnames(one_hot_encoding_test_df)



all_one_hot <- rbind(one_hot_encoding_train_df, one_hot_encoding_test_df)


ind_train_oh <- 1:nrow(one_hot_encoding_train_df)
ind_test_oh <- (nrow(one_hot_encoding_train_df) + 1):nrow(all_one_hot)

rotated_pca_one_hot <- prcomp(all_one_hot)
z <- kde2d(rotated_pca_one_hot$x[,1], rotated_pca_one_hot$x[,2], n = 500)
plot(rotated_pca_one_hot$x[ind_test_oh,1], rotated_pca_one_hot$x[ind_test_oh,2])
points(rotated_pca_one_hot$x[ind_train_oh,1], rotated_pca_one_hot$x[ind_train_oh,2], col="red")


rt = Rtsne(all_one_hot, verbose=T, perplexity=500)
plot(rt$Y[ind_test_oh,])
points(rt$Y[ind_train_oh,], col="red")
z <- kde2d(rt$Y[,1], rt$Y[,2], n = 100)

all_mt = read.csv("/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning/data/misc/random_100k_seq_emb.csv")
