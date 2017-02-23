model_dim = 25
chunk_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 24, 25, 50, 100, 500]
chunk_nums = map(s -> s < model_dim ? div(model_dim, s) : 1, chunk_sizes)
hmc_times = [26.9, 32.1, 22.3, 19.7, 16.7, 16.1, 15.0, 14.1, 14.3, 14.0, 14.0, 10.8, 10.9, 10.9, 11.7, 10.9, 10.7]

# using Plots
# plotlyjs()
# N = length(chunk_sizes)
# plot(chunk_sizes[1:N], chunk_nums[1:N], lab="Number of chunks")
# plot!(chunk_sizes[1:N], hmc_times[1:N], lab="Elapsed time (s)",
#       xaxis="Chunk size", yaxis="Number of chunks / elapsed time (s)")

using Gadfly

function draw_plot(N)
  plot(
    layer(
      x=chunk_sizes[1:N], y=chunk_nums[1:N], Geom.line,
      Theme(default_color=colorant"green")),
    layer(
      x=chunk_sizes[1:N], y=hmc_times[1:N], Geom.line,
      Theme(default_color=colorant"orange")
    ),
    Guide.XLabel("Chunk size"),
    Guide.YLabel("Number of chunks / elapsed time (s)"),
    Guide.Title("Evaluation of AD performance with chunk size varying"),
    Guide.manual_color_key("Legend", ["Number of chunks", "Elapsed time (s)"], ["green", "orange"]))
end

N = length(chunk_sizes)

plot1 = draw_plot(N)

N = length(chunk_sizes) - 2

plot2 = draw_plot(N)

draw(PDF("chunk-size-evaluation-1-500.pdf", 10inch, 5.625inch), plot1)
draw(PDF("chunk-size-evaluation-1-50.pdf", 10inch, 5.625inch), plot2)
