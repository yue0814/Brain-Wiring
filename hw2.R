## include libraries available from  CRAN only
t <- proc.time()
if (require(data.table) != T) {
        install.packages("data.table", repos="https://cloud.r-project.org/")
}
if (require(parallel) != T) {install.packages("parallel", repos="https://cloud.r-project.org/")}
library(parallel)
library(data.table)
print("Package loaded successfully.")
## Include the names of all team members
authors <- function() {
    c("Ludan Zhang", "Jiachen Zhang")
}
authors()
##1
#build up parallel
n.cores <- detectCores()
cl <- makeCluster(n.cores)
clusterEvalQ(cl, library(data.table)) #call library in the clusters

#The file should be in the same directory as the data
#get the names of all the files.
files <- dir(getwd())
files <- files[grepl(".txt",files)]
#Function to read the data
m.read <- function(file.name) {
        d <- fread(file.name, sep = " ")
        return(d)
}
#Function to do the z-transform
z.trans <- function(d) {
        cor.table <- cor(d)
        trans.table <- 0.5*(log(1 + cor.table) - log(1 - cor.table))
        trans.table[which(trans.table == Inf)] = 0
        return(trans.table)
}
#Read data and do the transform
m.s <- parLapply(cl, files, m.read)
f.s <- parLapply(cl, m.s, z.trans)

##2
#Function to calculate the statistic
m.calc <- function(x,f) {
        x1 <- array(unlist(x), dim = c(15,15,length(x)))
        return(apply(x1, c(1,2), f))
}
#Calculate mean and variance
f.n <- m.calc(f.s, mean)
f.v <- m.calc(f.s, var)

##3
#Separate the data into train and test set
f.s2 <- f.s[order(files)]
#Calculate the mean for train and test set
f.s.train.n <- m.calc(f.s2[1:410], mean)
f.s.test.n <- m.calc(f.s2[411:820], mean)

##4
m.s <- m.s[order(files)]
#Function to do the normalization
m.norm <- function(m) {
        return(as.data.frame(scale(m, center = colMeans(m))))
}
#Normalize the data
norm.m.s <- parLapply(cl, m.s, m.norm)
#Separate normalized data into train and test set
x.train <- rbindlist(norm.m.s[1:410])
x.test <- rbindlist(norm.m.s[411:820])
#SVD and other computation
u.g <- svd(x.train)
u <- u.g$u
g <- diag(u.g$d)%*%t(u.g$v)
c.ug <- cov(u%*%g)
c.train <- cov(x.train)
c.test <- cov(x.test)
c.ug.c.test <- norm(c.ug-c.test, type = "F")
c.train.c.test <- norm(c.train-c.test, type = "F")
#Function to save files
m.save <- function(name, data.list) {
        fwrite(data.table(data.list[[name]]), paste0(name,".csv"), sep = ',', row.names = F,col.names = F)
        return(paste0(name," saved successfully!"))
}
#Save all the files
m.all <- list(f.n,f.v,f.s.train.n,f.s.test.n,u,g,c.ug,c.train,c.test,c.ug.c.test,c.train.c.test)
names(m.all) <- c("Fn","Fv","Ftrain","Ftest","U","G","CUG","Ctrain","Ctest","CUGCtest","CtrainCtest") 
parLapply(cl,names(m.all),m.save,data.list = m.all)
#Compute the running time and stop the parallel cluster
print(proc.time() - t)
stopCluster(cl)
