% function POD_GalROM_SF(run,n,snp_dom)
clear
% CREATE GALERKIN ROM (POD) % STATIC FRAME of REFERENCE
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
% Code is for 3D pipe flow from the Open Pipe Flow (OPF) sim. 
% Inputs:
%       u  fluctuations [Nr,Nth,Nz,m]
%       v  fluctuations [Nr,Nth,Nz,m]
%       w  fluctuations [Nr,Nth,Nz,m]
%       r  vector
%       th vector
%       z  vector
%       Re (D,U_m)
%       dt (sim. timestep)
%       SR (sim. save rate)
%       Data location (path from 'pipe_flow' onwards)
%       Save location (path from 'pipe_flow' onwards)
% Outputs:
%       u_r_ROM [Nr,Nth,Nz,m] (total vel)
%       u_t_ROM [Nr,Nth,Nz,m] (total vel)
%       u_z_ROM [Nr,Nth,Nz,m] (total vel)
%       theta [N,n] (spatial modes)
%       c [t,n] (temporal coefficients)
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
clc

%% INPUT PARAMETERS -- MUST BE INSERTED!!!
% 
% Make sure PATH & PARAMETERS are correct.
% 
% ~~~~~~~~~~~~~~ ROM PARAMETERS ~~~~~~~~~~~~~~
n       = 8;         % Order of reduced order model (rank of projection) 
snp_dom = 1:128; % Which snapshots of data to take
run     = 'run0';   % Which run to take
decomp  = 2;         % 1 SVD || 2 POD
shift   = 0;         % 0 No shift mode || 1 Add shift mode


% ~~~~~~~~~~~~~~ Sim PARAMETERS ~~~~~~~~~~~~~~
Re     = 2200;      % Re number
alpha  = 0.0625;    % L = pi/alpha
dt_OPF = 0.01;      % Simulation dt
SR_OPF = 50;        % OPF save rate
SR     = 200;       % Desired save rate
dt     = dt_OPF*SR; % Sim time increment
SR_ratio = round(SR/SR_OPF);

% ~~~~~~~~~~~~~~ Save Location ~~~~~~~~~~~~~~~
ID = strcat(sprintf('%04d_%04d',snp_dom(1),snp_dom(end)), ...
    '_n',sprintf('%02d',n),'_shf',num2str(shift),'_FD');
% ID = snp(1)_snp(2)_n#_shf#_Dr# , n#, shift(0|1), FD (Full Domain)
% % *****************************************
% IP = 'JVdr';
% save_file = strcat('in_prod/',ID,'_',IP,'/'); 
% % *****************************************
save_file = strcat(run,'/',ID,'/');
save_loc = strcat("/home/shaul/OK/ROMs/",save_file);
status = rmdir(save_loc,'s');
mkdir(save_loc)

% ~~~~~~~~~~~~~~ Path Locations ~~~~~~~~~~~~~~
addpath("/home/shaul/OK/pipe_flow/Functions/")
addpath("/home/shaul/OK/pipe_flow/Post_Processing/")
addpath("/home/shaul/OK/pipe_flow/Post_Processing/Plots/")
addpath("/home/shaul/OK/pipe_flow/Post_Processing/Sections/")
addpath(strcat("/home/shaul/OK/Runs/",run))
addpath(strcat(save_loc))

% ~~~~~~~ Initialize Record Parameters ~~~~~~~
fig_n = 1;
log_name = strcat(save_loc,'Log');
delete(log_name)
diary (log_name)

%% SIMULATION PARAMETERS
fprintf('Loading data\n')

[u_r,u_t,u_z,u_mean,r,th,z] = ...
    sim_data_SF(snp_dom,run,alpha,SR_ratio,dt);

fprintf('\nData loaded\n')

% !!! CHECK THAT ALL MEANS ARE CORRECT !!!
% u_m_r & u_m_t should be identically 0 over a long (and long) enough time
% interval/ensemble. Use as a test to make sure 'm' is large enough.
u_m_z_lam = 1-r.^2;
u_m_r         = u_mean(:,:,:,1); % u_r velocity mean field
u_m_t         = u_mean(:,:,:,2); % u_t velocity mean field
u_m_z_fluc    = u_mean(:,:,:,3);
u_m_z = u_m_z_fluc + u_m_z_lam;  % u_z velocity mean field

% The visualization:
figure(fig_n)
subplot(3,1,1), pipe_sec(u_m_z_fluc,'u''_{z,avg}',r,th,z,1,0)
subplot(3,1,2), pipe_sec(u_m_r,'u_{r,avg}',r,th,z,1,0)
subplot(3,1,3), pipe_sec(u_m_t,'u_{t,avg}',r,th,z,1,0)
saveas(fig_n,strcat(save_loc,'pipe_sec_mean_vel.fig'))
close(fig_n), fig_n = fig_n + 1;

z_ind = [-4 -2 0 2];
figure(fig_n), cross_sec_puff(u_m_z_fluc,'u''_{z,avg}',r,th,z,z_ind)
saveas(fig_n,strcat(save_loc,'cross_sec_u_z.fig'))
close(fig_n), fig_n = fig_n + 1;
figure(fig_n), cross_sec_puff(u_m_r,'u_{r,avg}',r,th,z,z_ind)
saveas(fig_n,strcat(save_loc,'cross_sec_u_r.fig'))
close(fig_n), fig_n = fig_n + 1;
figure(fig_n), cross_sec_puff(u_m_t,'u_{t,avg}',r,th,z,z_ind)
saveas(fig_n,strcat(save_loc,'cross_sec_u_t.fig'))
close(fig_n), fig_n = fig_n + 1;

%% INITIAL DATA PROCESSING
m     = size(u_z,4);          % Number of snapshots
t_max = (m-1)*dt;             % Max time
t_sim = linspace(0,t_max,m)'; % Time discretization
Nr    = length(r);            % Number of x points
Nt    = length(th);           % Number of y points
Nz    = length(z);            % Number of z points
N     = Nr*Nt*Nz;             % Number of total grid points
N_tz  = Nt*Nz;

r_m1 = repmat(r.^(-1),N_tz,1); % 1/r
r_m2 = repmat(r.^(-2),N_tz,1); % 1/r^2

J  = repmat(r,N_tz,1); % Jacobian for inner product

%% SNAPSHOT MATRICES
% A is the fluctuation snapshot matrix
A = [reshape(u_r,N,m)
     reshape(u_t,N,m)
     reshape(u_z,N,m)];

%% POD & ROM
fprintf('\nBeginning POD procedure\n')

switch decomp
    case 1
        fprintf("\nSVD algorithm\n")
        % ~~~~~~~~~~~~~ SVD ~~~~~~~~~~~~~
        [U,S,V] = svd(A,"econ");

        theta = U(:,1:n); % Take first 'n' spatial modes
        psi = theta(1:N,:);       % u_r
        phi = theta(N+1:2*N,:);   % u_t
        xi  = theta(2*N+1:end,:); % u_z

    case 2
        fprintf("\nPOD algorithm\n")
        % ~~~~~~~~~~~~~ POD ~~~~~~~~~~~~~
        [lambda,c_sim_POD,psi,phi,xi] = POD(m,n,Nr,Nt,Nz,A);
        S = sqrt(lambda);

        psi = reshape(psi,N,[]);
        phi = reshape(phi,N,[]);
        xi  = reshape(xi ,N,[]);
        % ~~~~~ Still need to check the difference between them ~~~~~
        % 
        % !!! Initial test shows exact match. Recommend to check other
        % variouys cases to be sure !!!
        % 
        % Notice that using c_sim_POD is problematic since I added a shift
        % mode. An option is to still use them, just add the first
        % coefficient from the inverse projection.
        %
        % c_sim = c_sim_POD;
        % c_sim = (theta'*A)';
        % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
end
figure(fig_n), plot_sing_val(S)
saveas(fig_n,strcat(save_loc,'sing_val.fig'))
close(fig_n), fig_n = fig_n + 1;

% % ~~~~~~~~~~ Shift Modes ~~~~~~~~~~
switch shift
    case 0
        fprintf("\nNo shift modes incorporated\n")
        theta = [psi;phi;xi];
    case 1
        fprintf("\nShift modes incorporated\n")
        PSI = reshape(psi,Nr,Nt,Nz,[]);
        PHI = reshape(phi,Nr,Nt,Nz,[]);
        XI  = reshape(xi ,Nr,Nt,Nz,[]);
        THETA = [PSI;PHI;XI];
        u_del_a = [u_m_r
            u_m_t
            u_m_z - (1-r.^2)];
        u_del_b = u_del_a;
        for i = 1:n
            u_del_b = u_del_b - u_del_a.*THETA(:,:,:,i).^2;
        end
        u_del = u_del_b./sqrt(sum(u_del_b.^2,"all"));

        psi_shift = u_del(1:Nr      ,:,:);
        phi_shift = u_del(Nr+1:2*Nr ,:,:);
        xi_shift  = u_del(2*Nr+1:end,:,:);

        psi_shift = reshape(psi_shift,N,[]);
        phi_shift = reshape(phi_shift,N,[]);
        xi_shift  = reshape(xi_shift ,N,[]);

        psi = [psi_shift,psi];
        phi = [phi_shift,phi];
        xi  = [xi_shift ,xi ];
        theta = [psi;phi;xi];

        n = n+1; % Adding the shift mode into the truncation count
end
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

% c_sim = c_sim_POD;
c_sim = (theta'*A)';
fprintf('\nPOD procedure finished\n')

%% Visualize modes
nlev = 40; n_modes = 4;
z_ind = [-4 -2 0 2];
figure(fig_n), pipe_sec_modes(xi ,'Xi' ,r,th,z,1,n_modes,nlev)
saveas(fig_n,strcat(save_loc,'pipe_sec_xi.fig'))
close(fig_n), fig_n = fig_n + 1;
figure(fig_n), pipe_sec_modes(psi,'Psi',r,th,z,1,n_modes,nlev)
saveas(fig_n,strcat(save_loc,'pipe_sec_psi.fig'))
close(fig_n), fig_n = fig_n + 1;
figure(fig_n), pipe_sec_modes(phi,'Phi',r,th,z,1,n_modes,nlev)
saveas(fig_n,strcat(save_loc,'pipe_sec_phi.fig'))
close(fig_n), fig_n = fig_n + 1;
figure(fig_n), cross_sec_modes(xi ,'Xi' ,r,th,z,z_ind,n_modes,nlev)
saveas(fig_n,strcat(save_loc,'cross_sec_xi.fig'))
close(fig_n), fig_n = fig_n + 1;
figure(fig_n), cross_sec_modes(psi,'Psi',r,th,z,z_ind,n_modes,nlev)
saveas(fig_n,strcat(save_loc,'cross_sec_psi.fig'))
close(fig_n), fig_n = fig_n + 1;
figure(fig_n), cross_sec_modes(phi,'Phi',r,th,z,z_ind,n_modes,nlev)
saveas(fig_n,strcat(save_loc,'cross_sec_phi.fig'))
close(fig_n), fig_n = fig_n + 1;

%% SPATIAL DERIVATIVES
% * CONCERNING OPERATIONS w/ ANY FORM OF THE DISCRETE RADIAL POINTS 'r' %
% Notice that anytime I multiply by 'r', a restructured vector must be made
% to comply with the reshaping of the velocity field.
% ** I created these vectors (r_1,r_m1,r_m2). Make sure that they are
% located where they are needed.
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fprintf('\nBeginning differentiations\n')
[theta_dr,theta_dt,theta_dz,theta_lap,psi_dr,psi_dt,psi_dz,phi_dr, ...
    phi_dt,phi_dz,xi_dr,xi_dt,xi_dz,U_m_zdr,U_m_zdt,U_m_zdz,U_m_z_lap] = ...
    derivatives(n,r,r_m1,r_m2,th,z,Nr,Nt,Nz,N,theta,psi,phi,xi,u_m_z);

U_m_r = reshape(u_m_r,N,1);  % u_r velocity mean field
U_m_t = reshape(u_m_t,N,1);  % u_t velocity mean field
U_m_z = reshape(u_m_z,N,1);  % u_z velocity mean field
fprintf('\nDifferentiations finished\n')

%% NS EQ.
u_lam = reshape(repmat(u_m_z_lam,Nt,Nz),N,[]);

fprintf('\nBeginning matrix assignment\n')
[D_m,M_m,D_f,D_e,M_mf,M_dmf,M_f,M_e,g] = NSE(n,J,r_m1,r_m2, ...
    theta,theta_dr,theta_dt,theta_dz,theta_lap,psi,psi_dt, ...
    phi,phi_dt,xi,U_m_z,U_m_zdr,U_m_zdt,U_m_zdz,U_m_z_lap,u_lam);
fprintf('\nMatrix assignment finished\n')

%% ODE
fprintf('\nBeginning ODE procedure\n')
% Transformation of IC
c0 = c_sim(1,:);

% ODE options
% options = odeset('RelTol',1e-8,'AbsTol',...
%     1e-8*ones(1,size(c0,1))); % Set tolerance for integrator
options = [];
t_ROM = t_sim(1):0.2*dt/SR_ratio:t_sim(25);
[t_ROM,c_ROM] = ode23(@Gal_Proj,t_ROM,c0,options, ...
    D_m,M_m,D_f,D_e,M_mf,M_dmf,M_f,M_e,Re,n);
fprintf('\nODE integration done\n')

% ORGANIZE RESULTS
% u_r_ROM = U_m_z + psi*c_ROM';
% u_t_ROM = U_m_t + phi*c_ROM';
% u_z_ROM = U_m_z + xi *c_ROM';
% u_r_ROM = reshape(u_r_ROM,Nr,Nt,Nz,[]);
% u_t_ROM = reshape(u_t_ROM,Nr,Nt,Nz,[]);
% u_z_ROM = reshape(u_z_ROM,Nr,Nt,Nz,[]);

% MISCELLANEOUS PLOTS
% Generalize all functions
fig_n = coef_plot(c_ROM,c_sim,n,8,t_sim,t_ROM,shift,fig_n,save_loc);

% [l2_u_r,l2_u_t,l2_u_z] = ...
%     flow_field_comp(u_r,u_t,u_z,u_r_ROM,u_t_ROM,u_z_ROM);
% figure(fig_n), cross_sec_puff(l2_u_r,'l_2(u_r)',r,th,z,z_ind)
% saveas(fig_n,strcat(save_loc,'l2_ur_cross_sec.fig')), close(fig_n)
% fig_n = fig_n + 1;
% figure(fig_n), cross_sec_puff(l2_u_t,'l_2(u_t)',r,th,z,z_ind)
% saveas(fig_n,strcat(save_loc,'l2_ut_cross_sec.fig')), close(fig_n)
% fig_n = fig_n + 1;
% figure(fig_n), cross_sec_puff(l2_u_z,'l_2(u_z)',r,th,z,z_ind)
% saveas(fig_n,strcat(save_loc,'l2_uz_cross_sec.fig')), close(fig_n)
% fig_n = fig_n + 1;
% 
% div = DIVERGENCE(u_r_ROM,u_t_ROM,u_z_ROM,r,th,z);
% fprintf('\nDivergence of ROM at each snapshot is:\n')
% disp(div)

% phase_space_shift(c_ROM,c_sim,shift,4)
% VORTICITY(u_r_ROM,u_t_ROM,u_z_ROM,r,th,z)

%% SAVE DATA
fprintf('\nSaving data...\n')

save(strcat(save_loc,'ROM.m'),"theta","c_ROM","c_sim");
save(strcat(save_loc,'plant.mat'),"D_m","D_f","D_e",...
    "M_m","M_mf","M_dmf","M_f","M_e","g","c_ROM")

fprintf('\nData saved\n')


fprintf('\nROM procedure finished\n')
diary off