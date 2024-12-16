%% Project: Calibration of an Accelerometer Using GPS Measurements
% MATLAB script for simulating the system, implementing a Kalman filter,
% and performing Monte Carlo simulations.
clear; close all; clc;

%% Parameters and Initial Conditions
omega = 0.2; % rad/sec
sampling_time_acc = 0.005; % seconds (200 Hz)
sampling_time_gps = 0.2; % seconds (5 Hz)
mean_w = 0;
var_w = 0.0004; % process noise variance
mean_ba = 0;
var_ba = 0.01; % accelerometer bias variance
mean_v0 = 100; % m/s
var_v0 = 1; % m/s^2
mean_p0 = 0; % meters
var_p0 = 10^2; % meters^2
a = 10; % m/s^2
num_samples_acc = 30 / sampling_time_acc + 1; % Number of samples over 30 sec

%% System Simulation
% True acceleration, velocity, and position
t_acc = 0:sampling_time_acc:30;
a_t = a * sin(omega * t_acc);

% Integrate to calculate true model
v0 = mean_v0 + randn * sqrt(var_v0);
p0 = mean_p0 + randn * sqrt(var_p0);
v_t = v0 + a / omega - a / omega * cos(omega*t_acc);
p_t = p0 + (v0 + a / omega) * t_acc - a / (omega^2) * sin(omega*t_acc);

%% Accelerometer and GPS Measurements
% Parameters
bias = sqrt(var_ba) * randn; % m/s^2
w = sqrt(var_w) * randn(1, num_samples_acc); % process noise (m/s^2)
a_c = a_t + bias + w;
v_c = zeros(1,num_samples_acc);
p_c = zeros(1, num_samples_acc);
v_c(1) = mean_v0;
p_c(1) = mean_p0;
v_e = zeros(1, num_samples_acc);
p_e = zeros(1, num_samples_acc);
v_e(1) = v0;
p_e(1) = p0;

% Derive the velocity and position with Euler integration formula
for k = 2:num_samples_acc
    % Accelerometer model
    v_c(k) = v_c(k-1) + a_c(k-1) * sampling_time_acc;
    p_c(k) = p_c(k-1) + v_c(k-1) * sampling_time_acc + a_c(k-1) * sampling_time_acc^2 / 2;

    % Dynamic model
    v_e(k) = v_e(k-1) + a_t(k-1) * sampling_time_acc;
    p_e(k) = p_e(k-1) + v_e(k-1) * sampling_time_acc + a_t(k-1) * sampling_time_acc^2 / 2;
end

% GPS measurement
num_samples_gps = 30 / sampling_time_gps + 1;
t_gps = 0:sampling_time_gps:30;
V = diag([1, 0.0016]); % measurement noises covariance
eta = [sqrt(V(1,1)) * randn(size(t_gps));
    sqrt(V(2,2)) * randn(size(t_gps))];
z = [p_t(1:sampling_time_gps / sampling_time_acc:end);
    v_t(1:sampling_time_gps / sampling_time_acc:end)] + eta;
d_z = [p_t(1:sampling_time_gps / sampling_time_acc:end) - p_c(1:sampling_time_gps / sampling_time_acc:end);
    v_t(1:sampling_time_gps / sampling_time_acc:end) - v_c(1:sampling_time_gps / sampling_time_acc:end)] + eta;
% gps_position_measurements = p_t(t_gps) + sqrt(var_p0) * randn(size(t_gps));
% gps_velocity_measurements = v_t(t_gps) + sqrt(var_v0) * randn(size(t_gps));

%% Monte Carlo Simulation & Kalman Filter Estimation
% System matrices
phi = [1 sampling_time_acc -0.5 * sampling_time_acc^2;
    0 1 -sampling_time_acc;
    0 0 1];
gamma = -[0.5 * sampling_time_acc^2; sampling_time_acc; 0];
W = var_w; % process noise covariance

% Measurement matrices
H = [1 0 0; 0 1 0];

N_ave = 1000; % Number of realizations for Monte Carlo simulation

v_c_realization = zeros(1, num_samples_acc);
p_c_realization = zeros(1, num_samples_acc);
v_c_realization(1) = mean_v0;
p_c_realization(1) = mean_p0;

r_realization = zeros(2, num_samples_acc, N_ave);

e_p_pri = zeros(1, num_samples_acc, N_ave); % a priori position estimation error
e_v_pri = zeros(1, num_samples_acc, N_ave);
e_b_pri = zeros(1, num_samples_acc, N_ave);

e_p_post = zeros(1, num_samples_gps, N_ave); % a posteriori position estimation error
e_v_post = zeros(1, num_samples_gps, N_ave);
e_b_post = zeros(1, num_samples_gps, N_ave);

e_ave = zeros(3, num_samples_gps);
P_ave = zeros(3, 3, num_samples_gps);
r_corr = zeros(2, 2, num_samples_gps);
r_tm = zeros(size(d_z, 1), 1);

for realization = 1:N_ave
    % Generate new noise, bias, and measurements for each realization
    b_realization = sqrt(var_ba) * randn;
    w_realization = sqrt(var_w) * randn(1, num_samples_acc);
    a_c_realization = a_t + b_realization + w_realization;
    eta_realization = [sqrt(V(1,1)) * randn(size(t_gps));
                       sqrt(V(2,2)) * randn(size(t_gps))];

    for k = 2:num_samples_acc

        v_c_realization(k) = v_c_realization(k-1) + a_c_realization(k-1) * sampling_time_acc;
        p_c_realization(k) = p_c_realization(k-1) + v_c_realization(k-1) * sampling_time_acc + a_c_realization(k-1) * sampling_time_acc^2 / 2;
    end

    % Store recalculated v_c and p_c for this realization
    d_z_realization = [p_t(1:sampling_time_gps / sampling_time_acc:end) - p_c_realization(1:sampling_time_gps / sampling_time_acc:end);
                       v_t(1:sampling_time_gps / sampling_time_acc:end) - v_c_realization(1:sampling_time_gps / sampling_time_acc:end)] + eta_realization;

    % Kalman filter for each realization
    x_pri = zeros(3, num_samples_acc);
    M = zeros(3, 3, num_samples_acc);
    x_post = zeros(3, num_samples_acc);
    P = zeros(3, 3, num_samples_acc);
    
    % Initial data
    x_pri(:,1) = [0; 0; mean_ba];

    M_0 = diag([var_p0, var_v0, var_ba]);

    M(:,:,1) = M_0;

    S_0 = H * M_0 * H.' + V;
    K_0 = M_0 * H.' / S_0;
    r_0 = d_z_realization(:,1) - H * x_pri(:,1);
    ortho = zeros(3, num_samples_gps);

    P(:,:,1) = inv(inv(M_0) + H.'* inv(V) * H);

    x_post(:,1) = x_pri(:,1) + K_0 * r_0;

    for k = 2:num_samples_acc
        % Prediction step (propagation)
        x_pri(:,k) = phi * x_post(:,k-1);
        M(:,:,k) = phi * P(:,:,k-1) * phi' + gamma * W * gamma';

        if mod(k, sampling_time_gps/sampling_time_acc) == 1
            u = ceil(k/40); % GPS measurement index (5 Hz rate)
            % Measurement update
            r = d_z_realization(:,u) - H * x_pri(:,k); % residual
            S = H * M(:,:,k) * H' + V;
            K = M(:,:,k) * H.' / S; % Kalmarn gain
            x_post(:,k) = x_pri(:,k) + K * r;
            P(:,:,k) = (eye(size(K,1)) - K * H) * M(:,:,k);

            r_realization(:, u, realization) = r;
            r_corr(:,:,u) = r_corr(:,:,u) + r * r_tm';

            r_tm = r;

        else
            % No GPS measurement available: prediction only
            x_post(:,k) = x_pri(:,k);
            P(:,:,k) = M(:,:,k);

        end
    end

    e_p_pri(:,:,realization) = p_t - p_c_realization - x_pri(1,:);
    e_v_pri(:,:,realization) = v_t - v_c_realization - x_pri(2,:);
    e_b_pri(:,:,realization) = b_realization - x_pri(3,:);

    e_p_post(:,:,realization) = p_t(1:40:end) - p_c_realization(1:40:end) - x_post(1,1:40:end);
    e_v_post(:,:,realization) = v_t(1:40:end) - v_c_realization(1:40:end) - x_post(2,1:40:end);
    e_b_post(:,:,realization) = b_realization - x_post(3,1:40:end);

    e_realization = [e_p_post(:,:,realization); e_v_post(:,:,realization); e_b_post(:,:,realization)];
    e_ave = e_ave + e_realization;

    % Update ensemble covariance
    for k = 1:num_samples_gps
        P_ave(:,:,k) = P_ave(:,:,k) + (e_realization(:,k) - e_ave(:,k)) * (e_realization(:,k) - e_ave(:,k)).';
    end

    % Orthogonality Check
    for k = 1:num_samples_gps
        error_diff = e_realization(:,k) - e_ave(:,k);
        ortho(:,k) = ortho(:,k) + error_diff .* x_pri(:,k);
    end

end

% Normalize ensemble averages and covariances
e_ave = e_ave / N_ave;
P_ave = P_ave / (N_ave - 1);
r_corr = r_corr / N_ave;
ortho = ortho / N_ave;

%% Plot Results
% First tile: Position
figure;
tiledlayout(3, 1, 'TileSpacing', 'Compact', 'Padding', 'Compact');

nexttile;
plot(t_acc, p_t, 'LineWidth', 2);
hold on;
plot(t_acc, x_post(1,:) + p_c_realization, 'LineWidth', 2); hold off;

title('Position', 'FontSize', 16, 'Interpreter', 'latex');
ylabel('Position (m)', 'FontSize', 14, 'Interpreter', 'latex');
grid on;

% Create a single legend under the title
legend({'True', 'Estimate'}, ...
    'FontSize', 16, ...
    'Location', 'northoutside', ...
    'Orientation', 'horizontal', ...
    'Interpreter', 'latex', ...
    'Box', 'on');

% Second tile: Velocity
nexttile;
plot(t_acc, v_t, 'LineWidth', 2);
hold on;
plot(t_acc, x_post(2,:) + v_c_realization, 'LineWidth', 2);

title('Velocity', 'FontSize', 16, 'Interpreter', 'latex');
ylabel('Velocity (m/s)', 'FontSize', 14, 'Interpreter', 'latex');
grid on;
hold off;

% Third tile: Bias
nexttile;
plot(t_acc, b_realization * ones(size(t_acc)), 'LineWidth', 2);
hold on;
plot(t_acc, x_post(3,:), 'LineWidth', 2);

title('Bias', 'FontSize', 16, 'Interpreter', 'latex');
xlabel('Time (sec)', 'FontSize', 14, 'Interpreter', 'latex');
ylabel('Bias (m/s$^2$)', 'FontSize', 14, 'Interpreter', 'latex');
grid on;
hold off;

sgtitle('Position, Velocity, and Bias', 'FontSize', 20, 'Interpreter', 'latex');
set(gcf,'Position',[100,200,800,600]);

saveas(gcf, 'Kalman Filter.png');



figure;
tiledlayout(3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

% Position Errors
nexttile;
plot(t_acc, e_p_pri(:,:,N_ave), 'LineWidth', 1.5); hold on;
plot(t_gps, e_p_post(:,:,N_ave), 'LineWidth', 1.5);
plot(t_acc, squeeze(sqrt(P(1,1,:))), 'k:', 'LineWidth', 1.5);
plot(t_acc, -squeeze(sqrt(P(1,1,:))), 'k:', 'LineWidth', 1.5); hold off;
ylim([-1 1]);
title('Position Errors', 'FontSize', 16, 'Interpreter', 'latex');
ylabel('Position Error (m)', 'FontSize', 14, 'Interpreter', 'latex');
grid on;

legend('A priori','A posteriori', '$\sigma$',...
    'Interpreter','latex', ...
    'Orientation','horizontal',...
    'Location','northoutside', ...
    'FontSize',16);

% Velocity Errors
nexttile;
plot(t_acc, e_v_pri(:,:,N_ave), 'LineWidth', 1.5); hold on;
plot(t_gps, e_v_post(:,:,N_ave), 'LineWidth', 1.5);
plot(t_acc, squeeze(sqrt(P(2,2,:))), 'k:', 'LineWidth', 1.5);
plot(t_acc, -squeeze(sqrt(P(2,2,:))), 'k:', 'LineWidth', 1.5); hold off;
ylim([-0.3 0.3]);
title('Velocity Errors', 'FontSize', 16, 'Interpreter', 'latex');
ylabel('Bias Error (m/s)', 'FontSize', 14, 'Interpreter', 'latex');
grid on;

% Bias Errors
nexttile;
plot(t_acc, e_b_pri(:,:,N_ave), 'LineWidth', 1.5); hold on;
plot(t_gps, e_b_post(:,:,N_ave), 'LineWidth', 1.5);
plot(t_acc, squeeze(sqrt(P(3,3,:))), 'k:', 'LineWidth', 1.5);
plot(t_acc, -squeeze(sqrt(P(3,3,:))), 'k:', 'LineWidth', 1.5); hold off;
title('Bias Errors', 'FontSize', 16, 'Interpreter', 'latex');
xlabel('Time (s)', 'FontSize', 14, 'Interpreter', 'latex');
ylabel('Bias Error (m/s$^2$)', 'FontSize', 14, 'Interpreter', 'latex');
grid on;

sgtitle('A Priori and A Posteriori Errors', 'FontSize', 20, 'Interpreter', 'latex');

set(gcf,'Position',[100,200,800,600]);

saveas(gcf, 'Monte Carlo.png');

% e_ave
figure;
plot(t_gps, e_ave(1,:), 'LineWidth', 1.5, 'DisplayName', '$e^{pl}(t_{i})$'); hold on;
plot(t_gps, e_ave(2,:), 'LineWidth', 1.5, 'DisplayName', '$e^{vl}(t_{i})$');
plot(t_gps, e_ave(3,:), 'LineWidth', 1.5, 'DisplayName', '$e^{bl}(t_{i})$'); hold off;
title('The Ensemble Average of $e^{l}(t_{i})$', 'FontSize', 20, 'Interpreter', 'latex');
xlabel('Time (s)', 'FontSize', 16, 'Interpreter', 'latex');
ylabel('Error', 'FontSize', 16, 'Interpreter', 'latex');
legend('Location', 'best', 'Interpreter', 'latex', 'FontSize', 14);
grid on;

set(gcf,'Position',[100,100,600,400]);

saveas(gcf, 'e_ave.png');

% Compare Variances
figure;
t = tiledlayout(3, 3, 'TileSpacing', 'Compact', 'Padding', 'Compact');
legend_added = false; % Flag to ensure legend is added only once
for i = 1:3
    for j = 1:3
        nexttile;
        p1 = plot(t_gps, squeeze(P_ave(i,j,:)), 'LineWidth', 2);
        hold on;
        p2 = plot(t_gps, squeeze(P(i,j,1:40:end)), 'LineWidth', 2);
        hold off;
        grid on;
        title(['$P_{ave}($', num2str(i), ',', num2str(j), ') vs P(', num2str(i), ',', num2str(j), ')'], ...
            'Interpreter', 'latex', 'FontSize', 14);
      
        % Add legend only once
        if ~legend_added && i == 1 && j == 2
            lgd = legend([p1, p2], 'Ensemble', 'Kalman', ...
                'Location', 'northoutside', ...
                'Interpreter', 'latex', ...
                'Orientation', 'horizontal', ...
                'FontSize', 16);
            legend_added = true;
        end
    end
end

sgtitle('Ensemble vs Kalman Variance (All Elements)', ...
    'Interpreter', 'latex', 'FontSize', 20);

% Add global labels for the entire figure
xlabel(t, 'Time (s)', 'Interpreter', 'latex', 'FontSize', 18);
ylabel(t, 'Variance', 'Interpreter', 'latex', 'FontSize', 18);

set(gcf,'Position',[100,200,1200,800]);

saveas(gcf, 'Var Comparison.png');

% Compare Variances (All Differences in One Plot)
figure;
tiledlayout(1, 1, 'TileSpacing', 'Compact', 'Padding', 'Compact');

hold on;
for i = 1:3
    for j = 1:3
        diff = squeeze(P_ave(i,j,:) - P(i,j,1:40:end)); % Compute difference
        plot(t_gps, diff, 'LineWidth', 2, ...
            'DisplayName', ['$P_{ave}($', num2str(i), ',', num2str(j), ') - P(', num2str(i), ',', num2str(j), ')']);
    end
end
hold off;

% Add grid and labels
ylim([-100,300]);
grid on;
xlabel('Time (s)', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('Variance Difference', 'Interpreter', 'latex', 'FontSize', 16);
title('Difference Between Ensemble and Kalman Variance', 'FontSize', 20, 'Interpreter', 'latex');

lgd = legend('show', ...
    'Location', 'eastoutside', ...
    'Interpreter', 'latex', ...
    'Orientation', 'vertical', ...
    'FontSize', 16, ...
    'Interpreter', 'latex');

set(gcf, 'Position', [100, 200, 1000, 400]);

saveas(gcf, 'Var Difference Single Plot.png');


% Plot Orthogonality Results
figure;
plot(t_gps, ortho(1, :), 'LineWidth', 2, 'DisplayName', 'Position');
hold on;
plot(t_gps, ortho(2, :), 'LineWidth', 2, 'DisplayName', 'Velocity');
plot(t_gps, ortho(3, :), 'LineWidth', 2, 'DisplayName', 'Bias');
hold off;
grid on;
legend('Location', 'best', 'Interpreter', 'latex', 'FontSize', 14);
title('Orthogonality Check for the error in estimates with the estimate', 'Interpreter', 'latex', 'FontSize', 20);
xlabel('Time (s)', 'FontSize', 18, 'Interpreter', 'latex');
ylabel('Inner Product Value', 'FontSize', 18, 'Interpreter', 'latex');

set(gcf,'Position',[100,200,800,600]);

saveas(gcf, 'Orthogonality.png');

% Verify Residual Independence
figure;
imagesc(r_corr(:,:,end)); % Correlation matrix for residuals
colorbar;
title('Residual Correlation Matrix', 'Interpreter', 'latex', 'FontSize', 20);
xlabel('Measurement Index', 'Interpreter', 'latex', 'FontSize', 18);
ylabel('Measurement Index', 'Interpreter', 'latex', 'FontSize', 18);

set(gcf,'Position',[100,200,800,600]);

saveas(gcf, 'Residual.png');

% Plot Determinant of r_corr Over Time
figure;
tiledlayout(1, 1, 'TileSpacing', 'Compact', 'Padding', 'Compact');

% Initialize determinant array
det_r_corr = zeros(1, size(r_corr, 3));

% Compute determinants for all time steps
for u = 1:size(r_corr, 3)
    det_r_corr(u) = det(r_corr(:,:,u));
end

plot(t_gps, det_r_corr, 'LineWidth', 2);
grid on;

ylim([-2.5e-4, 2.5e-4]);
xlabel('Time (s)', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('Determinant of $r_{corr}$', 'Interpreter', 'latex', 'FontSize', 16);
title('Determinant of Residual Correlation Matrix Over Time', 'Interpreter', 'latex', 'FontSize', 20);

set(gcf, 'Position', [100, 200, 800, 400]);

saveas(gcf, 'Determinant_r_corr.png');