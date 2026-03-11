function [flow_out] = mask_flow(flow_in, mask)

flow_in_x = flow_in.Vx;
flow_in_y = flow_in.Vy;

flow_in_x(~mask)= 0;    % Zero invalid regions
flow_in_y(~mask)= 0;    % Zero invalid regions

flow_out = opticalFlow(flow_in_x, flow_in_y);