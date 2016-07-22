function adjust_figure()
for i=1:4
    fig=subplot(1,4,i);
    xlim([0.5 6.5]);
    set(fig,'FontSize',14);
    objects=fig.Children;
    for j=1:length(objects)
        set(objects(j), 'LineWidth', 2);
    end
end
end