%%
clc
clear
close all

%% true model
xit = [0.25 0.50 -0.75];
syms x(t)
eqn = diff(x,t) == xit(1)*x+xit(2)*sin(2*pi*t)+xit(3)*cos(2*pi*t);
cond = x(0) == 0.60;
xt = dsolve(eqn,cond);

%% main process
nset = [501,251,101];       % sample size
nvrs = [0.25,0.10,0.05];    % noise level

ttyp = 2;                   % type of time instants
res = nan(1,6);

for i=1:length(nset)
    nobs = nset(i);         % sample size    
    for j=1:length(nvrs)
        nvr = nvrs(j);      % noise-variance ratio
        
        %% data generation
        switch ttyp
            case 1                          % equally-spaced
                t = linspace(0,5,nobs)';
            otherwise                       % irregularly-spaced
                rng(2);                     % reproducible random numbers
                t = sort(5*rand(nobs,1));
        end      
        xtru = eval(subs(xt,t));            % true time series
        rng(2);                             % for reproducible
        nois = sqrt(nvr)*std(xtru)*randn(size(xtru));
        xobs = xtru+nois;                   % noisy observations
        v = [sin(2*pi*t) cos(2*pi*t)];      % true inputs
        
        %% Gauss-Newton based parameter estimation
        % initial guess: integral matching estimates 
        dt = diff(t);
        Theta = [ cumsum(xobs(2:end,:)+xobs(1:end-1,:)).*dt/2,...
                  cumsum(v(2:end,1)+v(1:end-1,1)).*dt/2,...
                  cumsum(v(2:end,2)+v(1:end-1,2)).*dt/2,...
                  ones(length(t)-1,1) ];
        xi0 = (Theta\xobs(2:end))';
        % iterative process: Gauss-Newton algorithm
        [xi,~,resid,~,~,~,J] = lsqnonlin('case2opt',xi0,[],[],[],t,xobs,v);
        [ci,se] = nlparci(xi,resid,'jacobian',J);
        res = [res; [[nobs;nan;nan] [nvr;nan;nan] [xi0; xi; se']]];        
        
        %% fitted trajactory 
        xfit = xobs-resid;
        
        %% figure 
        h = figure(3*(i-1)+j);
        plot(t,xobs,'.g','MarkerSize',10); hold on         
        plot(t,xtru,'-b', 'linewidth',1.5); 
        plot(t,xfit,'--r', 'linewidth',1.5); hold off
        ylim([0 3])
        xlabel('$t$','interpreter','latex')
        ylabel('$x(t)$','interpreter','latex')
        title([['n=',num2str(nobs)],[' nvr=',num2str(nvr)]])
        legend({'noisy';'true';'fitted'},'location','northwest','fontsize',11)
        set(gca,'fontsize',12)
        set(gcf,'Position',[50 190 400 400])        
    end
end

%% parameter estimates and standard derivation
res






