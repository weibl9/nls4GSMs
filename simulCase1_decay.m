% Copyright 2020, All Rights Reserved
% Code by Baolei Wei
% For paper, "Parameter estimation for grey system models: 
%				a nonlinear least squares perspective"
% by Baolei Wei, Naiming Xie

%%
clc
clear
close all

%% true model
xit = [-0.75 3.5];
syms x(t)
eqn = diff(x,t) == xit(1)*x;
cond = x(0) == xit(2);
xt = dsolve(eqn,cond);   % closed-form solution: xi(2)*exp(xi(1)*(t-t(1)))

%% mian process
nset = [501,101,51];            % sample size
nvrs = [0.50,0.25,0.05];        % noise level

ttyp = 2;                       % type of time instants
res = nan(1,4);

for i=1:length(nset)
    nobs = nset(i);     % sample size    
    for j=1:length(nvrs)
        nvr = nvrs(j);  % noise-variance ratio
        
        %% data generation
        switch ttyp
            case 1                          % equally-spaced
                t = linspace(0,5,nobs)';
            otherwise                       % irregularly-spaced
                rng(1);                     % reproducible radom numbers
                t = sort(5*rand(nobs,1));
        end       
        xtru = eval(subs(xt,t));            % true time series 
        rng(2);                             % for reproducible
        nois = sqrt(nvr)*std(xtru)*randn(size(xtru));
        xobs = xtru+nois;                   % noisy observations

        %% Gauss-Newton based parameter estimation
        % initial guess: integral matching estimates 
        dt = diff(t);
        Theta = [cumsum(xobs(2:end,:)+xobs(1:end-1,:)).*dt/2,...
                 ones(length(t)-1,1)];
        xi0 = (Theta\xobs(2:end))';
        % iterative process: Gauss-Newton algorithm
        optf1 = @(xi)xi(2)*exp(xi(1)*(t-t(1)))-xobs;	% closed-form solution
        [xi,~,resid,~,~,~,J] = lsqnonlin(optf1,xi0);
        [ci,se] = nlparci(xi,resid,'jacobian',J);
        res = [res; [[nobs;nan;nan] [nvr;nan;nan] [xi0; xi; se']]];        
        
        %% fitted trajactory 
        xfit = xi(2)*exp( xi(1)*(t-t(1)) );
        
        %% figure 
        h = figure(3*(i-1)+j);
        plot(t,xobs,'.g','MarkerSize',10); hold on         
        plot(t,xtru,'-b', 'linewidth',1.5); 
        plot(t,xfit,'--r', 'linewidth',1.5); hold off
        ylim([-1 4])
        xlabel('$t$','interpreter','latex')
        ylabel('$x(t)$','interpreter','latex')
        title([['n=',num2str(nobs)],[' nvr=',num2str(nvr)]])
        legend({'noisy';'true';'fitted'},'location','northeast','fontsize',11)
        set(gca,'fontsize',12)
        set(gcf,'Position',[50 190 400 400])
    end
end

%% parameter estimates and standard derivation
res






