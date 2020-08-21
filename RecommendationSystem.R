##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier
#movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#                                           title = as.character(title),
#                                          genres = as.character(genres))
# if using R 4.0 or later
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#DEVELOPER CODE STARTS HERE#
#1st step - We need to define our loss function and split edx into training and test dsets
#1.1-RMSE -> That will calculate how good is our algorith comparing our predicted results
#with actual ones coming from the test dataset
#We will use root mean squared error as loss funtion
rmseLossFunc <- function(act_ratings,pred_ratings){sqrt(mean((act_ratings-pred_ratings)^2))}

#1.2-Create edx_train and edx_test datasets
# Test set will be 20% of edx ratings dataset
set.seed(13, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(13)` instead
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
edx_train <- edx[-test_index,]
temp <- edx[test_index,]

# Make sure userId and movieId in edx_test set are also in edx_train set
edx_test <- temp %>% 
  semi_join(edx_train, by = "movieId") %>%
  semi_join(edx_train, by = "userId")

# Add rows removed from edx_test set back into edx_train set
removed <- anti_join(temp, edx_test)
edx_train <- rbind(edx_train, removed)

rm(test_index,temp,removed)

#2nd step - Start creating our model and refining it
#2.1-Calculate mean of all ratings
all_mean <- mean(edx_train$rating)
rmse1_all_mean <- rmseLossFunc(edx_test$rating,all_mean)
#1.061172

#2.2-For this iteration, we are going to predict the rating of a movie as the mean of
#ratings given for all users that rated that movie (regardless of user or genre)
movie_avg <- edx_train %>% group_by(movieId) %>% summarize(avg_rating=mean(rating)-all_mean)
pred_movie_avg <- edx_test %>% left_join(movie_avg,by="movieId") %>% .$avg_rating+all_mean
rmse2_movie_avg <- rmseLossFunc(edx_test$rating,pred_movie_avg)
#RMSE is 0.943987 --> Needs to be improved

#2.3-Use regularization for movie rating
rmseMovieAvgReg <- function(lambda){
  movie_avg <- edx_train %>% group_by(movieId) %>% summarize(avg_rating=sum(rating-all_mean)/(NROW(rating)+lambda))
  predicted_dset <- edx_test %>% left_join(movie_avg,by="movieId") %>% .$avg_rating
  predicted_dset <- predicted_dset + all_mean
  rmseLossFunc(edx_test$rating,predicted_dset)}

#Need to look for the best lambda that minimizes RMSE
lambdas <- seq(0,1000,100)
rmses <- sapply(lambdas,rmseMovieAvgReg)
plot(lambdas,rmses)
lambdas <- seq(0,100,10)
rmses <- sapply(lambdas,rmseMovieAvgReg)
plot(lambdas,rmses)
lambdas <- seq(0,10,0.5)
rmses <- sapply(lambdas,rmseMovieAvgReg)
plot(lambdas,rmses)

#Predict regularized model
lambda1 = lambdas[which.min(rmses)]
movie_avg_reg <- edx_train %>% group_by(movieId) %>% summarize(avg_rating=sum(rating-all_mean)/(NROW(rating)+lambda1),n=NROW(rating))
pred_movie_avg_reg <- edx_test %>% left_join(movie_avg_reg,by="movieId") %>% .$avg_rating+all_mean
rmse3_movie_avg_reg <- rmseLossFunc(edx_test$rating,pred_movie_avg_reg)
#New RMSE is 0.943929 --> Needs to be improved

#3rd step - Add user effect, some users like everything, others don't
#For this iteration, we are going to calculate the user effect as the mean
#of the rating of the user for each movie minus the overall rating for that movie
user_avg <- edx_train %>% select(userId,movieId,rating) %>% left_join(movie_avg_reg,by="movieId") %>% mutate(user_movie_rat=rating-avg_rating-all_mean) %>% group_by(userId) %>% summarize(user_rat=mean(user_movie_rat))
pred_useref_avg <- edx_test %>% left_join(movie_avg,by="movieId") %>% left_join(user_avg,by="userId") %>% mutate(useref_rat=avg_rating+user_rat+all_mean) %>% .$useref_rat
rmse4_useref_avg <- rmseLossFunc(edx_test$rating,pred_useref_avg)
#RMSE is 0.866465 --> It has improved adding user effect, let's try to improve it more

#4th step - Instead of using overall user effect, let's try to define user
#effect per genre. People that like comedies tend to give a better rating to 
#comedies, people that like action movies the same for action movies...
#As many movies have more than one genre, the final rating will
#include the average of the user effects for all the genres in the movie

#4.1-Need to tranform the edx dataset adding the genres for each movie in separate columns
#1-Idenfify how many and which separate different genres do we have
movies_genres <- edx_train %>% distinct(movieId,genres)
genres <- as.vector(str_split(movies_genres$genres,"\\|",simplify = TRUE))
diff_genres <- unique(genres)
diff_genres <- sort(diff_genres)
#[1] ""                   "(no genres listed)" "Action"            
#[4] "Adventure"          "Animation"          "Children"          
#[7] "Comedy"             "Crime"              "Documentary"       
#[10] "Drama"              "Fantasy"            "Film-Noir"         
#[13] "Horror"             "IMAX"               "Musical"           
#[16] "Mystery"            "Romance"            "Sci-Fi"            
#[19] "Thriller"           "War"                "Western" 
#Remove 1st 2 genres as are not correct
genres_list <- diff_genres[-1:-2]

for (i in 1:length(genres_list)){
  aux <- ifelse(str_detect(movies_genres$genres,genres_list[i]),1,0)
  movies_genres = cbind(movies_genres,aux)
  names(movies_genres)[i+2] <- str_replace(genres_list[i],"-","")
}

#Add number of genres per movie
movies_genres$n_genres <- rowSums(movies_genres[,3:21], na.rm = TRUE)

#Calculate user average per genre
user_movie_genre <- edx_train %>% select(userId,movieId,rating,genres) %>% left_join(movie_avg_reg,by="movieId") %>% select(-n) %>% mutate(user_movie_rat=rating-avg_rating-all_mean)
for (i in 1:length(genres_list)){
  aux <- ifelse(str_detect(user_movie_genre$genres,genres_list[i]),user_movie_genre$user_movie_rat,NA)
  user_movie_genre = cbind(user_movie_genre,aux)
  names(user_movie_genre)[i+6] <- str_replace(genres_list[i],"-","")
}

user_avg_genre <- user_movie_genre %>% group_by(userId) %>% summarise_at(vars(Action:Western),~ mean(.x, na.rm = TRUE))

#For the genres that user has not view/rated any movie, we have a NA.
#As we don't have info if the user like or don't like that genre, let's update
#those NAs with the overall average for each user
user_avg_all_genre <- user_avg_genre %>% left_join(user_avg,by="userId") %>% mutate_at(vars(Action:Western),~coalesce(.x,user_rat))

#Calculate prediction
#Let's use some matrix operations
genre_count <- movies_genres %>% select(movieId,n_genres)
test_user_genres <- edx_test %>% left_join(user_avg_all_genre,by="userId") %>% left_join(genre_count,by="movieId")
test_movies <- edx_test %>% select(-genres) %>% left_join(movies_genres,by="movieId")
test_user_genres[,7:25] <- test_user_genres[,7:25]*test_movies[,7:25]

#Take into account that could be movies without genres defined
#For those ones, we are going to assign the simplified user average
predicted_user_genres <- test_user_genres %>% mutate(user_rating=ifelse(n_genres!=0,rowSums(.[7:25],na.rm=TRUE)/n_genres,user_rat)) %>% select(movieId,userId,title,genres,rating,user_rating)
predicted_user_genres <- predicted_user_genres %>% left_join(movie_avg_reg,by="movieId") %>% mutate(useref_rat=avg_rating+user_rating+all_mean)
pred_useref_genre <- predicted_user_genres$useref_rat
rmse5_useref_genre <- rmseLossFunc(edx_test$rating,pred_useref_genre)
#0.852652 

#5th step - Explore is there is any effect taking into account the year
#movies were released. Theory: classical movies that are watched in modern times
#is because are better films, hence will have better average
#5.1-Need to extract the year from the title first
movies <- edx_train %>% distinct(movieId,title)
movies_year <- movies %>% extract(title,c("year"),"(\\(\\d\\d\\d\\d\\))$",remove = FALSE) %>% mutate(year=str_replace_all(year,c("\\("="","\\)"=""))) %>% select(movieId,year)

movie_avg <- edx_train %>% group_by(movieId) %>% summarize(avg_rating=mean(rating),count=NROW(rating))
movie_avg_year <- movie_avg %>% left_join(movies_year,by="movieId")
avg_year <- movie_avg_year %>% group_by(year) %>% summarize(avg_rat_year=mean(avg_rating),reviews=sum(count))
avg_year %>% ggplot(aes(year,avg_rat_year))+ geom_point(color='blue') + xlab("Year") + scale_x_discrete(breaks=seq(1915,2008,10)) + ylab("Average rate") + ggtitle("Average movie rate per release year") + theme(plot.title = element_text(hjust = 0.5))
#In the graph we can see that very old movies have a higher averarge rating (usually between 3.3-3.7 in years 1915 to 1970)
#than new ones, around (3-3.2 from 1985 to 2005)


#5.2-Add year to the previous model, so we have movie, user and year effect
#In order to avoid memory problems and simplify: 
#Forget about genres for calculating year effect, use user_average, not per genre
#Implement model
#Adding year effect
year_avg <- edx_train %>% left_join(movies_year,by="movieId") %>% select(userId,movieId,year,rating) %>% left_join(movie_avg_reg,by="movieId") %>% left_join(user_avg,by="userId")
year_avg <- year_avg %>% mutate(year_rat = rating - avg_rating - user_rat - all_mean) %>% group_by(year) %>% summarize(avg_rat_year=mean(year_rat))

pred_movie_user_year <- edx_test %>% left_join(movies_year,by="movieId") %>% left_join(movie_avg_reg,by="movieId") %>% left_join(year_avg,by="year")
pred_movie_user_year=cbind(pred_movie_user_year,user_rating=predicted_user_genres$user_rating)
pred_movie_user_year <- pred_movie_user_year %>% mutate(final_rat=all_mean + avg_rating + user_rating + avg_rat_year) %>% .$final_rat
rmse6_movie_user_year <- rmseLossFunc(edx_test$rating,pred_movie_user_year)
#0.852325

#6th step - Check expected min and max values for ratings
#If there are values over or below them, round them to max and min
min_rat <- min(edx_train$rating)
#0.5
max_rat <- max(edx_train$rating)
#5
pred_final <- pred_movie_user_year
ind_over_maxrat <- which(pred_final>max_rat)
ind_below_minrat <- which(pred_final<min_rat)
pred_final[ind_over_maxrat]=max_rat
pred_final[ind_below_minrat]=min_rat
rmse7_final <- rmseLossFunc(edx_test$rating,pred_final)
#0.8519

#Let's print the table comparing results for each iteration
rmse_summary <- tibble(Method = "Average of all movies", RMSE = rmse1_all_mean)
rmse_summary <- bind_rows(rmse_summary,tibble(Method="Average for each different movie",RMSE = rmse2_movie_avg))
rmse_summary <- bind_rows(rmse_summary,tibble(Method="Average for each different movie with regularization",RMSE = rmse3_movie_avg_reg))
rmse_summary <- bind_rows(rmse_summary,tibble(Method="Regularized movie + user average effect",RMSE = rmse4_useref_avg))
rmse_summary <- bind_rows(rmse_summary,tibble(Method="Regularized movie + user avg effect per genre",RMSE = rmse5_useref_genre))
rmse_summary <- bind_rows(rmse_summary,tibble(Method="Regularized movie + user effect per genre + movie release year",RMSE = rmse6_movie_user_year))
rmse_summary <- bind_rows(rmse_summary,tibble(Method="Regul. movie + user per genre + release year + round max/min",RMSE = rmse7_final))
rmse_summary %>% knitr::kable()

#FINAL RMSE USING VALIDATION DATASET
validation_user_genres <- validation %>% left_join(user_avg_all_genre,by="userId") %>% left_join(genre_count,by="movieId")
validation_movies <- validation %>% select(-genres) %>% left_join(movies_genres,by="movieId")
validation_user_genres[,7:25] <- validation_user_genres[,7:25]*validation_movies[,7:25]

predicted_user_genres <- validation_user_genres %>% mutate(user_rating=rowSums(.[7:25],na.rm=TRUE)/n_genres) %>% select(movieId,userId,title,genres,rating,user_rating)
predicted_user_genres <- predicted_user_genres %>% left_join(movie_avg_reg,by="movieId") %>% mutate(useref_rat=avg_rating+user_rating+all_mean)
pred_useref_genre <- predicted_user_genres$useref_rat

ind_over_maxrat <- which(pred_useref_genre>max_rat)
ind_below_minrat <- which(pred_useref_genre<min_rat)
pred_final <- pred_useref_genre
pred_final[ind_over_maxrat]=max_rat
pred_final[ind_below_minrat]=min_rat

finalRmse <- rmseLossFunc(validation$rating,pred_final)
#0.8521
