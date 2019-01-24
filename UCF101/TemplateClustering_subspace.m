function [template,alignpath,sigmaw] = TemplateClustering_subspace(classnum,trainset,traindownset,trainsetnum,max_iteration_num_ini,template_length,band_factor)
    
    alignpath = cell(1,classnum);
    for c = 1:classnum
        alignpath{c} = cell(1,trainsetnum(c));
        for i = 1:trainsetnum(c)
            alignpath{c}{i} = zeros(template_length,2);
        end
    end
    
    %% ini alighpath_ini
    for c = 1:classnum
        for i = 1:trainsetnum(c)
            features = traindownset{c}{i};
            feanum = size(features,1);
            partemp_align_path = zeros(template_length,2);
            if feanum>=template_length
                partemp_ave_align_num = [floor(feanum/template_length) floor(feanum/template_length)];
                temp_start = 1;
                for temp_align_count = 1:template_length-1
                    temp_end = temp_start + partemp_ave_align_num(mod(temp_align_count,2)+1)-1;
                    partemp_align_path(temp_align_count,:) = [temp_start temp_end];
                    temp_start = temp_end + 1;
                end
                temp_end = feanum;
                partemp_align_path(template_length,:) = [temp_start temp_end];
            else
                partemp_ave_align_num = [max(floor(template_length/feanum),1) max(floor(template_length/feanum),1)];
                temp_start = 1;
                for temp_align_count = 1:feanum-1
                    temp_end = temp_start + partemp_ave_align_num(mod(temp_align_count,2)+1)-1;
                    for temp_tem_count = temp_start:temp_end
                        partemp_align_path(temp_tem_count,:) = [temp_align_count temp_align_count];
                    end
                    temp_start = temp_end + 1;
                end
                temp_end = template_length;
                temp_align_count = feanum;
                for temp_tem_count = temp_start:temp_end
                    partemp_align_path(temp_tem_count,:) = [temp_align_count temp_align_count];
                end
            end
            alignpath{c}{i} = partemp_align_path;
        end
    end
    
    %% Updating alignpath and template iteratively
    dim = size(features,2);
    template = cell(1,classnum);
    template_num = cell(1,classnum);
    for c = 1:classnum
        template{c} = zeros(template_length,dim);
        template_num{c} = zeros(1,template_length);
    end
    
    for ite = 1:max_iteration_num_ini
        template_old = template;
        for c = 1:classnum
            mean_sequence = zeros(template_length,dim);
            mean_sequence_num = zeros(1, template_length);
            for i = 1:trainsetnum(c)
                features = traindownset{c}{i};
                temp_align_path = alignpath{c}{i};
                for temp_align_count = 1:template_length
                    temp_start = temp_align_path(temp_align_count,1);
                    temp_end = temp_align_path(temp_align_count,2);
                    mean_sequence_num(temp_align_count) = mean_sequence_num(temp_align_count) + temp_end - temp_start + 1;
                    mean_sequence(temp_align_count,:) = mean_sequence(temp_align_count,:) + sum(features([temp_start:temp_end],:),1);             
                end           
            end
            for temp_align_count = 1:template_length
                if mean_sequence_num(temp_align_count) > 0
                    mean_sequence(temp_align_count,:) = mean_sequence(temp_align_count,:)./mean_sequence_num(temp_align_count);
                end
            end
            template{c} = mean_sequence;
            template_num{c} = mean_sequence_num;
        end
        
        flag = 0;
        for c = 1:classnum
            
            if sum(sum(abs(template{c}-template_old{c}))) > 10^(-6)
                flag = 1;
                for i = 1:trainsetnum(c)
                    features = traindownset{c}{i};
                    feanum = size(features,1);
                    if feanum>=template_length
                        [dis,temp_align_path] = computeWarpingPathtoTemplate_Eud_band_addc(features, template{c}, band_factor);
                        partemp_align_path = temp_align_path;
                    else
                        [dis,temp_align_path] = computeWarpingPathtoTemplate_Eud_band_addc(template{c}, features, band_factor);
                        for temp_align_count = 1:feanum
                            temp_start = temp_align_path(temp_align_count,1);
                            temp_end = temp_align_path(temp_align_count,2);
                            for temp_tem_count = temp_start:temp_end
                                partemp_align_path(temp_tem_count,:) = [temp_align_count temp_align_count];
                            end
                        end
                    end
                    alignpath{c}{i} = partemp_align_path;
                end
            end
        end
        
        if flag == 0
            ite
            break;
        end
    end
    
    %% calculate Sw and Sb
    dim = size(trainset{1}{1},2);
    template_full = cell(1,classnum);
    template_full_num = cell(1,classnum);
    for c = 1:classnum
        template_full{c} = zeros(template_length,dim);
        template_full_num{c} = zeros(1,template_length);
    end
    
    state_variance = cell(1,classnum);
    for c = 1:classnum
        state_variance{c} = cell(1,template_length);
        for temp_align_count = 1:template_length
            state_variance{c}{temp_align_count} = zeros(dim,dim);
        end
    end
    
    for c = 1:classnum
        mean_sequence = zeros(template_length,dim);
        mean_sequence_num = zeros(1, template_length);
        for i = 1:trainsetnum(c)
            features = trainset{c}{i};
            temp_align_path = alignpath{c}{i};
            for temp_align_count = 1:template_length
                temp_start = temp_align_path(temp_align_count,1);
                temp_end = temp_align_path(temp_align_count,2);
                mean_sequence_num(temp_align_count) = mean_sequence_num(temp_align_count) + temp_end - temp_start + 1;
                mean_sequence(temp_align_count,:) = mean_sequence(temp_align_count,:) + sum(features([temp_start:temp_end],:),1);             
            end           
        end
        for temp_align_count = 1:template_length
            if mean_sequence_num(temp_align_count) > 0
                mean_sequence(temp_align_count,:) = mean_sequence(temp_align_count,:)./mean_sequence_num(temp_align_count);
            end
        end
        template_full{c} = mean_sequence;
        template_full_num{c} = mean_sequence_num;
        
        
        for i = 1:trainsetnum(c)
            features = trainset{c}{i};
            temp_align_path = alignpath{c}{i};
            for temp_align_count = 1:template_length
                temp_start = temp_align_path(temp_align_count,1);
                temp_end = temp_align_path(temp_align_count,2);
                for temp_index_count = temp_start:temp_end
                    temp_vector = features(temp_index_count,:) - template_full{c}(temp_align_count,:);
                    state_variance{c}{temp_align_count} = state_variance{c}{temp_align_count} + temp_vector'*temp_vector;
                end
            end
        end
        for temp_align_count = 1:template_length
            if template_full_num{c}(temp_align_count)~=0
                state_variance{c}{temp_align_count} = state_variance{c}{temp_align_count}./template_full_num{c}(temp_align_count);
            end
        end
    end
    
    all_feature_num = 0;
    for c = 1:classnum
        all_feature_num = all_feature_num + sum(template_num{c});
    end
    state_prior = template_num;
    for c = 1:classnum
        state_prior{c} = state_prior{c}./all_feature_num;
    end
    
    sigmaw = zeros(dim,dim);
    for c = 1:classnum
        for temp_align_count = 1:template_length
            sigmaw = sigmaw + state_prior{c}(temp_align_count)*state_variance{c}{temp_align_count};
        end
    end
end