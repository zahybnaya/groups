library('ggplot2')
datafiles=list.files(pattern="*.csv$")
for (datafile in datafiles) {
    d=read.csv(datafile)
    print(datafile)
    maxGroups = max(d$groupsFound)
    d$recall=d$groupsFound/maxGroups
    d$resource<-d$requestsNum
    upToRequest<-0
    for (a in levels(d$alg)){
        maxResource = min(d[d$alg==a&d$groupsFound==maxGroups,]$requestsNum)
        upToRequest<-max(upToRequest,maxResource)
        d[d$alg==a,]$resource=d[d$alg==a,]$resource/maxResource
    }
    #print(ggplot(data=d, aes(x=d$requestsNum, y=d$recall, group=factor(d$alg))) +  geom_line(aes(linecolour=factor(d$alg))) + ggtitle("Finding Entire Bipartite graph") + ylab(" % groups") + xlab("#flips") )
    #      + theme(legend.position="top") + theme(legend.position="top"))+ scale_linetype_discrete(name="Method:") + theme_bw())
    #dev.off()

    d<-d[d$requestsNum<=upToRequest,]
    pdf(paste(paste("flips",datafile,sep=""),".pdf",sep=""))
    print(ggplot(data=d, aes(x=d$requestsNum, y=d$recall, col = factor(d$alg), linetype = factor(d$alg))) +
          geom_line(lwd = 1) +
          scale_linetype_discrete() +
          scale_color_discrete() +
          ggtitle("Finding Entire Bipartite graph") + ylab(" % groups") + xlab("# flips") + theme_bw()+ theme(legend.position="top")+theme(legend.text=element_text(size=4)))
    dev.off()

    d=d[d$resource<=1.0,]
    pdf(paste(paste("auc",datafile,sep=""),".pdf",sep=""))
    print(ggplot(data=d, aes(x=d$resource, y=d$recall, group=factor(d$alg))) +  geom_line(aes(linetype= factor(d$alg))) + geom_point(size=0) + ggtitle("Finding Entire Bipartite graph") + ylab(" % groups") + xlab("% total flips") + scale_linetype_discrete(name="Method:") + theme_bw()+ theme(legend.position="top") + theme(legend.position="top"))
    dev.off()
}

