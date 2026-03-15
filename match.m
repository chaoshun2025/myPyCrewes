function [mfilt,tm]=match(trin,trdsign,t,mlength,flag)
% [mfilt,tm]=match(trin,trdsign,t,mlength,flag)
%
% MATCH designs a match filter of temporal length 'mlength'
% which matches trin to trdsign in the least squares sense.
% That is sum_of_sqs(conv(mfilt,trin)-trdsign)==minimum
%
% trin= input trace to be matched to trdsign
% trdsign= input trace which is to be matched
% t= time coordinate vector for trin
% ***** note: trin and trdsign must be the same length
% mlength= length of the match filter in seconds
% flag=0 ... a noncausal operator is desired
%     =1 ... a causal operator is desired
% NOTE: Suppose that 'a' is a known time series and 'w' is also
%  a known wavelet. Then let bm=convm(a,w) and bz=convz(a,w).
%  Then [westm,tw]=match(a,bm,ta,length(w),1) or
%       [westz,tw]=match(a,bz,ta,length(w),0) 
%  will both produce good estimates of w. 
%  However, 
%       [westx,tw]=match(a,bz,ta,length(w),1) 
%  should not be used
%  but
%       [westy,tw]=match(a,bm,ta,2*length(w),0)
%  will produce a valid estimate of w in the second half of westy.         
%
% mfilt= output mlength match filter
% tm= time coordinate for the match filter
%
% Matlab source file.

% preliminaries
 n=round(mlength/(t(2)-t(1)))+1;
 trin2=trin(:);
 trdsign=trdsign(:);
% generate the Toeplitz matrix for the normal equations
 TRIN= convmtx(trin2,n);
% solve the equations with left division
 if flag==1
  mfilt=TRIN\[trdsign;zeros(n-1,1)];
  tm=xcoord(0.,t(2)-t(1),mfilt);
 else
  nh=fix(n/2);
  top=[zeros(nh,1);trdsign;zeros(n-nh-1,1)];
  mfilt=TRIN\top;
  tm=xcoord(-(nh)*(t(2)-t(1)),t(2)-t(1),mfilt);
 end
 [j,k]=size(trin);
 if j==1, mfilt=mfilt.'; tm=tm'; end
   
 
 
