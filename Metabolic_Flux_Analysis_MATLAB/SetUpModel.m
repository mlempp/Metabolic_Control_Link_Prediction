%Author: NF
%This code takes the standard flux distribution from the Maranas model 
%and re-adjusts it by using metabolic flux analysis, to get closer to a
%steady state.
%If needed, metabolites and reactions can be removed. We use the cobra
%toolbox here to modify the network.
rng(1)
%initCobraToolbox
load model
modelx = model;
clear model

%%%%% setup model structure %%%%%
rxns = modelx.rxn;
mets = modelx.metab;
model.S = modelx.S;
model.mets = mets;
model.rxns = rxns;
model.Vnet = modelx.Vss;
model.Sreg = modelx.enzymeReg;

%adjust stoichiometric matrix and fluxes according to flux direction
model1 = change_directions(model);
%remove metabolites
model2 = remove_mets(model1);

%remove isolated rxns
model3 = remove_isolated_rxns(model2);

%model4 = add_reg(60,model3);
[G,model] = construct_bipartite_graph(model3);

%add regulation
[met_id,rxn_id] = ind2sub(size(model.Sreg),find(model.Sreg));
mode = model.Sreg(find(model.Sreg));
reg_edges = [met_id,length(model.mets)+rxn_id,mode];

%%%%% remove substrate/product regulation %%%%%
[~,id1,id2] = intersect(G.Edges.EndNodes,reg_edges(:,1:2),'rows');
reg_edges(id2,:) = [];

Ereg_new = zeros(size(model.Sreg));
linearInd = sub2ind(size(Ereg_new), reg_edges(:,1), reg_edges(:,2)-length(model.mets));
Ereg_new(linearInd) = reg_edges(:,3);
model.Sreg = Ereg_new;
G = addedge(G,reg_edges(:,1),reg_edges(:,2),2*ones(length(reg_edges),1));


%remove non-connected modules/ isolated modules
[G,model] = remove_nonconnected_modules(model,G);

%%%how many target genes do the regulators have
idx = G.Edges.Weight == 2;
regedges = G.Edges.EndNodes(idx,:);
mets = regedges(:,1);
reactions = regedges(:,2);
unique_mets = unique(mets);
unique_rxn = unique(reactions);
for k=1:length(unique_mets)
    count(k) = sum(unique_mets(k)==mets); 
end
for k=1:length(unique_rxn)
    count1(k) = sum(unique_rxn(k)==reactions); 
end
figure(1)
subplot(1,2,1)
histogram(count)
subplot(1,2,2)
histogram(count1);


Nodes = [model.mets;model.rxns];%re-define nodes
Ereg = model.Sreg;

EnsembleSize = 10000;
cutoffs = [0.0005, 2/3];
CCC_results = cell(EnsembleSize,1);
FCC_results = cell(EnsembleSize,1);

Net.Vref = model.Vnet;
Net.EnzName = model.rxns;
Net.MetabName = model.mets;
Net.S = model.Sorig;
Net.Sreg = model.Sreg;
Net.Reversibilities = zeros(size(model.rxns));


for k = 1:EnsembleSize
        disp(k)
        E = setupE(model.Easo_scaffold,Ereg);
        jac = model.Sorig*diag(model.Vnet)*E';
        ew = max(real(eig(jac)));
        CCC = -1*pinv(model.Sorig*diag(model.Vnet)*E')*model.Sorig*diag(model.Vnet);
        FCC =  E'*CCC+diag(ones(length(model.Vnet),1));    
        CCC_results{k,1} = -1*CCC;
        FCC_results{k,1} = -1*FCC;
end

[categorized_data] = categorization(CCC_results, FCC_results, cutoffs);
[minmax_data,mean_data] = minmaxing(CCC_results, FCC_results);
categorized_native = abs(categorized_data).*mean_data;

%==============================================================================
%some more analyses

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%how many regulations can we realistically infar?
id = model.Vnet < 0.001;
fraction_low_flux = sum(id)/length(model.Vnet); %so roughly 1/3 of the fluxes have an insignificant flux
reg_low = model.Sreg(:,id);
reg_high = model.Sreg;
reg_high(:,id) = [];
total_reg = sum(model.Sreg(:) ~=0);
low_reg = sum(reg_low(:) ~=0)/total_reg;
high_reg = sum(reg_high(:) ~=0)/total_reg;


%
CCC = categorized_data(1:size(CCC_results{1},1),:);

state = cell(size(CCC,2),1);
for k = 1:size(CCC,2)
    sub_id = find(model4.S(:,k) == -1);
    sub_name = model4.mets(sub_id);
    state(k) = {max(CCC(sub_id,k))};
    if isempty(sub_name)
       state(k) = {2};
    end
end
state = cell2mat(state);
state(state == 2) = [];
fraction = sum(state==1)/length(state);
fraction1 = sum(state==-1)/length(state);

%fraction of reactions where at least 1 subsrate is upregulated upon perturbation
%bla = state == (model.Vnet>0.001); % in x % of cases the substrate is upregulated when the reaction carries a high flux (above 0.001)

disp(fraction) %fraction where at least 1 substrate is upregulated upon perturbation of the following reaction
%disp(sum(bla)/length(bla));

%list all regulating metabolites
%X = regulation_effects(model,CCC);

x = score_each_regulation(model,categorized_data,G,Nodes);
x_new = x~=0;

%%% histogram of maximum distance of each node
dists = max(distances(G));
figure(2)
hist(dists)
%%% distribution of the amount of connected neighbours / node degree
%%% distribution

degs = degree(G);
figure(3)
hist(degs,40)

CorrMat = Correlations(categorized_data,Net); %correlations between metabolites and reactions
CorrMat(isnan(CorrMat))=0;

blurr = CorrMat.*abs(Net.Sreg);
rest = CorrMat.*abs(~Net.Sreg);

scatter(1:length(rest(:)),rest(:),15,'filled');
hold on
scatter(1:length(blurr(:)),blurr(:),15,'r','filled');

%=============================================================================%















function CorrMat = Correlations(categorized_data,Net)
n_met = length(Net.MetabName);
n_rxn = length(Net.EnzName);

CCC = categorized_data(1:n_met,:);
FCC = categorized_data(1:n_rxn,:);

for k = 1:n_met
    for kk = 1:n_rxn  
    a = CCC(k,:);
    b = FCC(kk,:);
    %id = a ~= 0 & b ~= 0;
    %correlation(k,kk) = corr2(a(id),b(id));    
    correlation(k,kk) = corr2(a,b);
    end
end

CorrMat = correlation;
end

function [X] = regulation_effects(model,CCC)
for k = 1:size(model.Sreg,1)
    if isempty(find(model.Sreg(k,:))) == 1
    IDx(k) = 0;
    else
    IDx(k) = 1;
    end
end
Regulator_ids = find(IDx);
Regulator_names = model.mets(Regulator_ids);
%for every regulator, get metabolites that should be affected

counter = 1;
for k = 1:length(Regulator_ids)
    regulator = Regulator_names(k);
    regulator_id = Regulator_ids(k);
    
    reg_rxns = find(model.Sreg(regulator_id,:));
    %get subs and prod of those reactions
    for kk = 1:length(reg_rxns)
        rxn_idx = reg_rxns(kk);
        sub_ids = find(model.Sorig(:,rxn_idx) <0);
        sub_names = model.mets(sub_ids);
        prod_ids = find(model.Sorig(:,rxn_idx) >0);
        prod_names = model.mets(prod_ids);
        X(counter,1) = {regulator_id};
        X(counter,2) = {sub_ids'};
        X(counter,3) = {prod_ids'};
        counter = counter + 1;
    end
end

for k = 1:size(X,1)
    met = cell2mat(X(k,:));
    %where does the regulator change?
    
    condition_ids = find(CCC(met(1),:));%find conditions where the regulator changes
    cond_number(k) = length(condition_ids);
    for kk = 1:length(condition_ids)
    idss = condition_ids(kk);
    cond = CCC(:,idss);
    out(k,kk) = sum(abs(cond(met)));
    end 
end

for k = 1:size(out,1)
    Out(k) = sum(out(k,:)>1)/cond_number(k);

end

figure(5)
hist(Out)
end


function Scores_new = score_each_regulation(model,patterns,G,Nodes)

id = G.Edges.Weight == 2;
reg_edges = G.Edges.EndNodes(id,:);

reg_edges_score = []
%how often does the regulator change? and in which conditions?
counter = 0;
for k = 1:length(reg_edges)
    
    Reg_id = reg_edges(k,1);
    Rxn_id = reg_edges(k,2);
    Condition_id = find(patterns(Reg_id,:));
    SubstrateChanges = abs(patterns(Reg_id,Condition_id)); %number of regulator changes
    RxnChanges = abs(patterns(Rxn_id,Condition_id));
    SubRxnCoChanges(k,1) = sum(SubstrateChanges == RxnChanges);
    if isempty(SubstrateChanges)
       counter = counter+1;
    end
    %do the substrates/products of the target reaction change aswell?
    target_subs = find(model.Sorig(:,Rxn_id-length(model.mets)) == 1);
    target_prods = find(model.Sorig(:,Rxn_id-length(model.mets)) == -1);
    
    TargetSubsChanges = sum(abs(patterns(target_subs,Condition_id)),1) ~= 0;
    TargetProdChanges = sum(abs(patterns(target_prods,Condition_id)),1) ~= 0;
    Together = (TargetSubsChanges + TargetProdChanges) ~= 0;
    
    %how often does the regulator, the rxn and the substrates/products
    %change
    CoChanges = (RxnChanges + Together) == 2;
    Count = sum(CoChanges);
    reg_edges_score(k,1) = Count;
end

Scores = horzcat(SubRxnCoChanges, reg_edges_score);
Scores_new = zeros(length(id),2);
list = find(id);

for k = 1:length(list)
    pos = list(k)
    Scores_new(pos,:) = Scores(k,:);
end    
end



function model = change_directions(model)
        Vx = (model.Vnet>=0) - (model.Vnet<0);
        model.Vnet = model.Vnet.*Vx;
        model.S = model.S*diag(Vx);
end

function model = remove_mets(model)
METS_TO_REMOVE = {'h_c','h_e','h2o_e','o2_e','co2_e','pi_e', 'h2o_c', 'co2_c', 'o2_c', 'pi_c',...
    'coa_c', 'thf_c', '5mthf_c','5fthf_c', 'methf_c', 'mlthf_c',...
    'nh4_c', 'cmp_c', 'q8_c', 'q8h2_c','udp_c', 'udpg_c',...
    'fad_c','fadh2_c', 'ade_c', 'ctp_c', 'gtp_c', 'h2o2_c',...
    'mql8_c', 'mqn8_c', 'na1_c', 'ppi_c', 'ACP_c','atp_c', 'adp_c', 'amp_c',...
     'nad_c', 'nadh_c', 'nadp_c', 'nadph_c'};

model = removeMetabolites(model,METS_TO_REMOVE,false);
end

function model = remove_isolated_rxns(model)
for k = 1:size(model.S,2)
    if isempty(find(model.S(:,k))) == 1
        idr(k) = 1;
    else
        idr(k) = 0;
    end
end
model = removeRxns(model,model.rxns(logical(idr)),false);

end


function model = from_scratch_reg(num_reg, model)
    %%% create random Sreg with as minimal overlap as possible
    model.Sreg = zeros(size(model.Sreg)); %removes current regulation
    xmets = randsample(length(model.mets),length(model.mets));
    id_met = xmets(1:num_reg);
    %as a test scenario only regulate reaction with sufficient flux
    [vsort,idsort] = sort(model.Vnet,'descend');
    id_rxn = idsort(1:num_reg);

    for k = 1:num_reg
        model.Sreg(id_met(k),id_rxn(k)) = -1;
    end
end

function model = add_reg(num_reg, model)
Sreg = model.Sreg;

%find metabolites that are not yet involved in regulation
for k = 1:size(Sreg,1)
    if isempty(find(Sreg(k,:)))
        val = 1;
    else
        val = 0;
    end
    met_ids(k) = logical(val);
end
met_ids = find(met_ids);

%find reactions that are not yet involved in regulation
for k = 1:size(Sreg,2)
    if isempty(find(Sreg(:,k)))
        val = 1;
    else
        val = 0;
    end
    rxn_ids(k) = logical(val);
end
rxn_ids = find(rxn_ids);
significant_flux_ids = model.Vnet(rxn_ids) > 0.001;
rxn_ids = rxn_ids(significant_flux_ids);

%randomly choose regulator and reactions
metx = randsample(met_ids,num_reg,0);
rxnx = randsample(rxn_ids,num_reg,0);

for j = 1:num_reg
    mety = metx(j);
    rxny = rxnx(j);
    model.Sreg(mety,rxny) = -1; %negative regulation
    if j > 0.9*num_reg
    model.Sreg(mety,rxny) = -4;    
    end
end

end


function [G,model] = construct_bipartite_graph(model)
model.Easo_scaffold = model.S < 0;
model.Sorig = model.S;
model.S = model.S ~= 0;
m = size(model.S,1); %~met dimensions
n = size(model.S,2); % ~rxn dimensions

Snew = [zeros(m,m), model.S; %incidence matrix for bipartite graph, dim: cnum + bnum x cnum + bnum,  met + gene x met + gene
        model.S', zeros(n,n)];
    
G =  graph(Snew); %convert matrix to graph structure


%adjust Ereg
Sreg1 = model.Sreg >= -3 & model.Sreg <0; %negative regulation
Sreg2 = model.Sreg == -4; %positive regulation
model.Sreg = Sreg2 - Sreg1;

end

function [G,model] = remove_nonconnected_modules(model,G)
distx = distances(G);
inf_ids = find(isinf(distx(:,1))); 
G = rmnode(G,inf_ids);
Nodes = [model.mets;model.rxns];
Nodes2Remove = Nodes(inf_ids);

%remove them also in the model
met2remove = [];
rxn2remove = [];
for k = 1:length(Nodes2Remove)
    met_val = find(strcmp(Nodes2Remove{k},model.mets));
    if ~isempty(met_val)
        met2remove(k,1) = met_val;
    end
    rxn_val = find(strcmp(Nodes2Remove{k},model.rxns));
    
    if ~isempty(rxn_val)
        rxn2remove(k,1) = rxn_val;
    end
end

model = removeMetabolites(model,model.mets(met2remove),false);
model = removeRxns(model,model.rxns(rxn2remove(rxn2remove~=0)),false);
end

function [categorized] = categorization(CCC_results, FCC_results, cutoffs)
    ccc_x = cat(3, CCC_results{:});
    up = ccc_x > cutoffs(1);
    down = ccc_x < -1*cutoffs(1);
    up_fraction = sum(up,3)/size(up,3);
    down_fraction = sum(down,3)/size(down,3);
    
    up_final = up_fraction > cutoffs(2);
    down_final = down_fraction > cutoffs(2);
    categorical_ccc = up_final - down_final;
    
    fcc_x = cat(3, FCC_results{:});
    up = fcc_x > cutoffs(1);
    down = fcc_x < -1*cutoffs(1);
    up_fraction = sum(up,3)/size(up,3);
    down_fraction = sum(down,3)/size(down,3);
    
    up_final = up_fraction > cutoffs(2);
    down_final = down_fraction > cutoffs(2);
    categorical_fcc = up_final - down_final;
    
    categorized = vertcat(categorical_ccc,categorical_fcc);
end

function [E] = setupE(Easo_scaffold,Ereg)
    LB = log10(0.001); %define lower bound
    UB = log10(1); %define upper bound
    id_easo = find(Easo_scaffold);
    Easo1 = zeros(size(Easo_scaffold));
    Easo1(id_easo) = 10.^(LB+(UB-LB)*rand(1,length(id_easo)));    
    Easo = Easo1;
    LB = log10(1); %define lower bound
    UB = log10(4); %define upper bound
    Ereg(find(Ereg)) = Ereg(find(Ereg)).*10.^(LB+(UB-LB)*rand(1,length(find(Ereg))))';
    
    E = Easo + Ereg;
end


function [minmax_data, mean_data] = minmaxing(CCC_results,FCC_results)
CCC_mean = mean(cat(3, CCC_results{:}), 3);
FCC_mean = mean(cat(3, FCC_results{:}), 3);

for k=1:size(CCC_mean,1)
    CCC_new(k,:) = rescale(CCC_mean(k,:),-1,1);
end

for k=1:size(FCC_mean,1)
    FCC_new(k,:) = rescale(FCC_mean(k,:),-1,1);
end
minmax_data = [CCC_new;FCC_new];

mean_data = [CCC_mean;FCC_mean];
end