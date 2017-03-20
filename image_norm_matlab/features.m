%
% Determine the vector of features values for this image.
%
% Image is normalized to [0, 1] and mask(i,j) = true
% indicates the pixel should be excluded from the calculation.
%
function f = features( img, mask )

    f = zeros( 8, 1 );

    %
    % Feature f1 in Cheng-Prasad-Brown.
    % Chromaticity of the average RGB value (gray world algorithm).
    %
    data = reshape( img, [], 3 );
    %data( mask, : ) = [];

    c = mean( data );
    c = c ./ sum( c );

    f(1) = c(1);
    f(2) = c(2);

    %
    % Feature f2 in Cheng-Prasad-Brown.
    % Chromaticity of the color of the pixel which has
    % the largest brightness (R + G + B).
    %
    [cnt, map] = hist3D(reshape(data,1,[],3));
    map( cnt < 200, : ) = [];
    cnt( cnt < 200 ) = [];
    chrom_map( :, 2 ) = map( :, 2 ) ./ sum( map, 2 );
    chrom_map( :, 1 ) = map( :, 1 ) ./ sum( map, 2 );
    [~, idxMax] = max( sum( map, 2 ) );

    f(3) = chrom_map( idxMax, 1 );
    f(4) = chrom_map( idxMax, 2 );

    %
    % Feature f3 in Cheng-Prasad-Brown.
    % Chromaticity of the color of the pixels belonging
    % to a histogram bin which has the largest count.
    %
    [~, idxMax ] = max( cnt );

    f(5) = chrom_map(idxMax,1);
    f(6) = chrom_map(idxMax,2);

    %
    % Feature f4 in Cheng-Prasad-Brown.
    % Mode of the image color pallete in chromaticity space.
    %
    [~,density, r, g] = kde2d( chrom_map, 0.005, 2^8, [0 0], [1,1] );
    [~, idxPeak] = max( density(:) );

    f(7) = r(idxPeak);
    f(8) = g(idxPeak);

end

