
main.loop <- function(indicator.list, price.scale, variables.score, price.scale.train, variables.train,n.runs ) {
  
  
  MAPE.mtx = matrix(ncol=1)
  RMSE.mtx = matrix(ncol=1)
  
  mape.i <- 1
  
  MAPE.prev <- 200
  RMSE.prev <- 2
  
  indicator.allowed.mtx = matrix(nrow = length(indicator.list), ncol = n.runs)
  
  for (jj in 1:n.runs){
    indicator.allowed = sample(indicator.list)     # randomize the order of the indicators
    indicator.allowed.mtx[,jj] <- indicator.allowed
    
  }
  
  
  for (run.i in 1:n.runs) {  #main loop for finding best indicators
    indicator.best.mtx = matrix(ncol=1) # initialize the list of best values
    AIC.mtx = matrix(ncol=1)
    
    AIC.prev <- 0  # this controls the number of desired indicators
    i <- 1
    
    print("****************************************")
    print(paste("***********", "Run=", run.i , "***********"))
    print("****************************************")
    
    
    for(ii in indicator.allowed.mtx[,run.i]) {   # first we find the best indicators for each random sample (each run.i iteration)
      #other fitting methods has to be examined too
      catch <-  tryCatch({  fit.data <- lm(price.scale ~ variables.score[,ii])},   #linear regression model
                         error=function(e) e ) 

      #fit.data <- glm(price.scale ~ variables.score[,ii], family = gaussian(link = "identity")) #generalized linear models
      #fit.data <- nls(price.scale ~ variables.score[,ii], start = 1) #non-linear least square
      #fit.data <- gam(price.scale ~ variables.score[,ii]) #generalized additive models
      
      step <- stepAIC(fit.data, trace=FALSE)   #finding the AIC of each indicator
      anova.s <- step$anova # display results
      AIC <- anova.s$AIC[1]
      
      if(AIC < AIC.prev){
        indicator.best.mtx[[i]] <- (ii)
        AIC.mtx[[i]] <- AIC
        i<- i+1 
        AIC.prev <- AIC }
      
    } #end of ii loop
    
    
    for(p.i in 1:12){       #To find order p & q
      for(q.i in 1:6){
        
        catch <-  tryCatch({   
          fit.Arima = Arima(price.scale.train,order=c(p.i,1,q.i),include.constant=T, xreg = variables.train[,c(indicator.best.mtx)])},
          
          error=function(e) e ) 
        if(inherits(catch, "error")) next(p.i=0 , q.i=0)
        
        price.forecast.test <- forecast(fit.Arima, h=horizon.test, xreg = variables.score[l.train:l.score,c(indicator.best.mtx)])
        price.forecast.test.points <- data.frame(price.forecast.test)
        error <- abs(price.forecast.test.points[,1] - price.scale[l.train:l.score,])
        RMSE <- round(sqrt(mean(error^2)),5)
        percent.error <- 100*error/abs(price.scale[l.train:l.score,])
        MAPE <- round(mean(percent.error),1)
        mape.i <- mape.i+1
        
        
        if(RMSE < RMSE.prev){
          indicator.best = matrix(ncol=1) # initialize the list of best values
          indicator.best <- indicator.best.mtx
          AIC.best = matrix(ncol=1)
          AIC.best <- AIC.mtx
          num.indicators <- length(indicator.best)
          RMSE.prev <- RMSE
          MAPE.prev <- MAPE 
          order.pq <- c(p.i,1,q.i)
          indicator.best.order.pq <- rbind(c(p.i,q.i,num.indicators,indicator.best,AIC.best))
        }
        
        
        print(paste("RMSE=", RMSE ,"  MAPE=", MAPE,  "  p=", p.i , "  q=", q.i))
        
      } #end of q loop
      
    } #end of p loop
    
  } #end of run loop
  
  return(indicator.best.order.pq)
} #end of function
