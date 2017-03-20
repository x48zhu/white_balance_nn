function generate()

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Gehler-Shi dataset.
    %
    dataset{1} = 'Canon1D';
    dataset{2} = 'Canon5D';
     
    for i = 1:length( dataset )
        diary( [dataset{i} '.csv'] );
        generateGS( dataset{i} );
        diary off
    end

end

%
% For each image in a data set, generate a labeled training example.
%
% Data sets/cameras:
%       Canon1D
%       Canon5D
%
function generateGS( camera )

    imageDir = 'C:\Users\vanbeek\Home\Research\Archive\Benchmarks\ImagesWhiteBalance\Gehler-Shi\';

    %
    % Look up information on images: coordinates of color checker,
    % (unnormalized) RGB ground truth, black and saturation cutoffs.
    %
    if( camera=='Canon1D' );
        data = Canon1D();
    end
    if( camera=='Canon5D' );
        data = Canon5D();
    end
    nImages = length( data );

    %
    % For each image in the data set:
    %
    for i = 1:nImages
        groundTruth = [data(i).gt_r, data(i).gt_g, data(i).gt_b];

        %
        % Read in image and normalize to [0, 1].
        %
        img = double( imread( [imageDir camera '\PNG\' data(i).imageName] ) );
        img = (double( img ) - data(i).darkness_level) ...
                  / (data(i).saturation_level - data(i).darkness_level);
        img( img < 0 ) = 0;

        %
        % Create mask for saturated pixels and colour checker.
        %
        [m, n, o] = size( img );
        mask = false( m, n );
        threshold = 0.98;
        mask( img(:,:,1) >= threshold ) = true;
        mask( img(:,:,2) >= threshold ) = true;
        mask( img(:,:,3) >= threshold ) = true;
        coordinates = [data(i).cc1, data(i).cc2, data(i).cc3, data(i).cc4];
        coordinates = min( coordinates, [Inf size(mask,1) Inf size(mask,2)] );
        coordinates = max( coordinates, [1 1 1 1] );
        mask( coordinates(1):coordinates(2), coordinates(3):coordinates(4) ) = true;

        %
        % Black out saturated pixels and colour checker.
        %
	for j = 1:m
	    for k = 1:n
		if( mask( j, k ) )
		    img( j, k, 1 ) = 0;
		    img( j, k, 2 ) = 0;
		    img( j, k, 3 ) = 0;
		end
	    end
	end

	%
	% At this point the image can be resized.
	%

        %
        % Determine the feature values for this image.
        %
        f = features( img, mask );

        %
        % Write out the labeled training data for this image.
        %
        fprintf( '%0.8f, %0.8f', f(1), f(2) );
	for k = 3:size( f )
            fprintf( ', %0.8f', f(k) );
	end
	fprintf( ', %0.8f, %0.8f\n', groundTruth(1), groundTruth(2) );

    end
end

