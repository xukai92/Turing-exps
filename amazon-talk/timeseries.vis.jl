using Gadfly
Gadfly.push_theme(:dark)

plottimeseries(y, yt, fn) = begin
  p_layer = layer(x=1:N, y=y, Geom.point, Theme(default_color=colorant"royalblue"))
  l_layer = layer(x=1:N, y=yt, Geom.line)

  plt = plot(p_layer, l_layer);

  draw(PNG("$fn.png", 6inch, 4.5inch), plt)
end
