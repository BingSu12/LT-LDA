function [distance,path] = computeWarpingPathtoTemplate_Eud_band_addc(sequence_sample, template, band_factor)
num_frames = size(sequence_sample, 1);
template_length = size(template, 1);
max_local_length = num_frames-template_length+1;

band_size = band_factor*(num_frames/template_length);
max_local_length = max(ceil(band_size),1);

scores = ones(num_frames, template_length, max_local_length) * -10^(20);
factor = template_length/num_frames;
offset = zeros(num_frames, template_length);


%scores(1,1,1) = dot(sequence_sample(1,:), template(1,:))/num_frames + (template_d(1)*(factor-template_u(1)) + template_a(1)*((factor-template_u(1))^2))/template_length;
scores(1,1,1) = -norm((sequence_sample(1,:) - template(1,:)),2)^2;  % + (template_d(1)*(factor-template_u(1)) + template_a(1)*((factor-template_u(1))^2))/template_length;

for j = 1:template_length
    for i = j:num_frames
%         if abs(i-j)>band_size
%             for l=1:min(i-j+1,max_local_length)
%                 scores(i,j,l) = -100000;
%             end
%         else
            %c_ij = dot(sequence_sample(i,:), template(j,:))/num_frames;
            c_ij = -norm((sequence_sample(i,:) - template(j,:)),2)^2;
            for l = 1
                if j>1
                    %delta_jl = (template_d(j)*(factor*l-template_u(j)) + template_a(j)*((factor*l-template_u(j))^2))/template_length;
                    %scores(i,j,l) = c_ij+max(max(scores(i,j-1,:)),max(scores(i-1,j-1,:)))+delta_jl;
                    scores(i,j,l) = c_ij + max(scores(i-1,j-1,:));  % + delta_jl;
                    indextemp = find(scores(i-1,j-1,:)==max(scores(i-1,j-1,:)));
                    offset(i,j) = indextemp(1);
                end
            end
            for l = 2:min(i-j+1,max_local_length)
                %delta_jl = (template_d(j)*(factor*l-template_u(j)) + template_a(j)*((factor*l-template_u(j))^2))/template_length;
                %delta_jln = (template_d(j)*(factor*(l-1)-template_u(j)) + template_a(j)*((factor*(l-1)-template_u(j))^2))/template_length;
                scores(i,j,l) = c_ij + scores(i-1,j,l-1);   % + delta_jl - delta_jln;
            end
%        end
    end
end
distance = max(scores(num_frames,template_length,:));  % + template_c;
%ltemp = find(scores(num_frame,template_length,:)==distance);
match_frame = template_length;
f = num_frames;
%path = int16(zeros(num_frames, 1));
path = zeros(template_length,2);
maxvalue = max(scores(f,match_frame,:));
ltemp = find(scores(f,match_frame,:)==maxvalue);
ltemp = ltemp(1);

while (f>0)        
    path(match_frame,:) = [f-ltemp+1 f];
    %path(f-ltemp+1:f) = match_frame;
    f = f - ltemp;
%    if f>1
        match_frame = match_frame - 1;
%         if match_frame<=0
%             path(1,:) = [1 f];
%             f = 0;
%         else
            ltemp = offset(f+1,match_frame+1);
%             [temp1,index1] = max(scores(f,match_frame,:));
%             [temp2,index2] = max(scores(f+1,match_frame,:));
%             [maxvalue,maxindex] = max([temp1,temp2]);
%             if maxindex(1)==2
%                 f = f + 1;
%             end

%            maxvalue = max(scores(f,match_frame,:));
%            ltemp = find(scores(f,match_frame,:)==maxvalue);
%            ltemp = ltemp(end);
%        end
%    end
end

if f~=0
    f
    path
    disp('Not match!');
end
%path(1,:) = [1 f];
% if path(1,:)==[0 0]
%     if f~=1
%         disp('Not match!!');
%     end
%     path(1,:) = [1 1];
% end

end