using Revise
using StatsModels, MixedModels, DataFrames
#import Plots
using Logging
using unfold
logger = SimpleLogger(stdout, Logging.Debug)
old_logger = global_logger(logger) 

include("dev/unfold/test/test_utilities.jl"); # to load the simulated data



data, evts = loadtestdata("testcase6","dev/unfold/test/")
data = data.+ 0.5*randn(size(data)) # we have to add minimal noise, else mixed models crashes.

categorical!(evts,:subject);
categorical!(evts,:stimulus);
f  = @formula 0~1+condA+(1+condA|subject) + (1+condA|stimulus);

data_r = reshape(data,(1,:))
# cut the data into epochs
data_epochs,times = unfold.epoch(data=data_r,tbl=evts,τ=(-0.4,0.8),sfreq=100);
# missing or partially missing epochs are currenlty _only_ supported for non-mixed models!
evts,data_epochs = unfold.dropMissingEpochs(evts,data_epochs)
@time m,results = unfold.fit(UnfoldLinearMixedModel,f,evts,data_epochs,times) 

# 2.6s seconds
f2  = @formula 0~1+condA*stimulus;
@time m,results = unfold.fit(UnfoldLinearModel,f2,evts,data_epochs,times) 
if 1==0
    using AlgebraOfGraphics, AbstractPlotting,WGLMakie
m = mapping(:colname_basis,:estimate,color=:group,layout_x=:term,layout_y=:group)
AlgebraOfGraphics.data(results) * visual(Lines) * m  |> draw

end

