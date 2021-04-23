using Random, Plots
using ForwardDiff
using OrdinaryDiffEq, DiffEqSensitivity
using LinearAlgebra
using Statistics
using ProgressBars, Printf
using Flux
using Flux.Optimise:update!
using Flux.Losses: mae, mse
using BSON: @save, @load
using DelimitedFiles
using YAML
using XLSX
using StatsBase

ENV["GKSwstype"] = "100"

conf = YAML.load_file("./config.yaml")

fuel_name = conf["fuel_name"]
expr_name = conf["expr_name"]
fig_path = string("./results/$fuel_name/$expr_name/figs")
ckpt_path = string("./results/$fuel_name/$expr_name/checkpoint")
config_path = "./results/$fuel_name/$expr_name/config.yaml"

is_restart = Bool(conf["is_restart"])
ns = Int64(conf["ns"])
nr = Int64(conf["nr"])
lb = Float64(conf["lb"])
n_epoch = Int64(conf["n_epoch"])
n_plot = Int64(conf["n_plot"])
grad_max = Float64(conf["grad_max"])
maxiters = Int64(conf["maxiters"])
n_train = Int64(conf["n_train"])
batchsize = Int64(conf["batchsize"])

lr_max = Float64(conf["lr_max"])
lr_min = Float64(conf["lr_min"])
lr_decay = Float64(conf["lr_decay"])
lr_decay_step = Int64(conf["lr_decay_step"])
w_decay = Float64(conf["w_decay"])

llb = lb;
global p_cutoff = -1.0

opt = Flux.Optimiser(
    ExpDecay(lr_max, lr_decay, n_train * lr_decay_step, lr_min),
    ADAMW(0.005, (0.9, 0.999), w_decay),
);

if !is_restart
    if ispath(fig_path)
        rm(fig_path, recursive=true)
    end
    if ispath(ckpt_path)
        rm(ckpt_path, recursive=true)
    end
end

if ispath("./results/$fuel_name") == false
    mkdir("./results/$fuel_name")
end

if ispath("./results/$fuel_name/$expr_name") == false
    mkdir("./results/$fuel_name/$expr_name")
end

if ispath(fig_path) == false
    mkdir(fig_path)
    mkdir(string(fig_path, "/conditions"))
end

if ispath(ckpt_path) == false
    mkdir(ckpt_path)
end

cp("./config.yaml", config_path; force=true)
