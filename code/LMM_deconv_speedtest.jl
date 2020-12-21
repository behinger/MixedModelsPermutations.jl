using Revise
using StatsModels, MixedModels, DataFrames
#import Plots
using Logging
using unfold
logger = SimpleLogger(stdout, Logging.Debug)
old_logger = global_logger(logger) 

include("../dev/unfold/test/test_utilities.jl"); # to load the simulated data



data, evts = loadtestdata("testcase6","dev/unfold/test/")
categorical!(evts,:subject);
categorical!(evts,:stimulus);
evts.subjectB = evts.subject;
evts1 = evts[evts.condA.==0,:]
evts2 = evts[evts.condA.==1,:]
data = data.+ 0.5*randn(size(data)) # we have to add minimal noise, else mixed models crashes.


f1  = @formula 0~1+condB+(1|subject)
f2  = @formula 0~1+condB+(1|subjectB)


data_r = reshape(data,(1,:))
# cut the data into epochs
#data_epochs,times = unfold.epoch(data=data_r,tbl=evts,τ=(-0.4,0.8),sfreq=100);
# missing or partially missing epochs are currenlty _only_ supported for non-mixed models!
#evts,data_epochs = unfold.dropMissingEpochs(evts,data_epochs)

ba1 = firbasis(τ=(0,1),sfreq = 100,name="evts")
ba2 = firbasis(τ=(0,1),sfreq = 100,name="evts")
#basisfunction2 = firbasis(τ=(0,0.5),sfreq = 10,name="basis2")
X1  = designmatrix(UnfoldLinearMixedModel,f1,evts1,ba1)
X2  = designmatrix(UnfoldLinearMixedModel,f2,evts2,ba2)
m = unfold.unfoldfit(UnfoldLinearMixedModel,X1+X2,data_r);
#@time m,results = unfold.fit(UnfoldLinearMixedModel,f,evts,data_r,basisfunction1) 
if 1==0
    using AlgebraOfGraphics, AbstractPlotting,WGLMakie
m = mapping(:colname_basis,:estimate,color=:group,layout_x=:term,layout_y=:group)
AlgebraOfGraphics.data(results) * visual(Lines) * m  |> draw

end

