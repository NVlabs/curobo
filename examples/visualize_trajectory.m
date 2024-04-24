close all; clear; clc;

% obs_num = 40;
world_id = 5;

robot = importrobot('kinova_gen3_7dof.urdf');
robot.DataFormat = 'col';

% load([num2str(obs_num), 'obs/world_',num2str(world_id-1),'.mat']);
load(['hard/world_',num2str(world_id),'.mat']);

% load(['curobo_trajectory_',num2str(obs_num),'_',num2str(world_id-1),'.mat']);
load(['curobo_trajectory_hard_',num2str(world_id),'.mat']);
q = double(q);

if length(size(q)) == 3
    q = squeeze(q(2,:,:));
end

figure;
show(robot, q(1,:)', 'Frames', 'off');

hold on;
for j = 1:size(obstacle_size,1)
    plot_box(obstacle_pos(j,:)', obstacle_size(j,:)');
end

% obs_num = 40;
% successes = zeros(1,100);
% for world_id = 1:100
% load(['curobo_trajectory_',num2str(obs_num),'_',num2str(world_id-1),'.mat']);
% successes(world_id) = if_success;
% end
% disp(sum(successes))

for i = 1:5:size(q,1)
    show(robot, q(i,:)');
end

axis off;
box on;

function h = plot_box(center, size)
    Z = zonotope(center, diag(size / 2));

    v = vertices(Z)';

    v_conv_hull = convhulln(v);
    h = trisurf(v_conv_hull,v(:,1),v(:,2),v(:,3), ...
                        'FaceColor',[1,0,0], ...
                        'FaceAlpha',0.5, ... 
                        'EdgeColor',[1,0,0], ...
                        'EdgeAlpha',0.5);
end
