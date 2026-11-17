// New local video export only. Original footage and tracking data are read-only.
#import <Foundation/Foundation.h>
#import <AVFoundation/AVFoundation.h>
#import <QuartzCore/QuartzCore.h>
#import <AppKit/AppKit.h>
#include <math.h>

static const int W=1920,H=900,FPS=25;
static NSString *assetDirectory;
static NSMutableDictionary *imageCache;

static id image(NSString *name) {
    id cached=imageCache[name];if (cached)return cached;
    NSImage *im=[[NSImage alloc] initWithContentsOfFile:[assetDirectory stringByAppendingPathComponent:name]];
    CGImageRef cg=[im CGImageForProposedRect:NULL context:nil hints:nil];
    if (!cg){fprintf(stderr,"Missing image %s\n",name.UTF8String);exit(10);}
    id value=(__bridge id)cg;imageCache[name]=value;return value;
}

static BOOL number(id value) {return [value isKindOfClass:NSNumber.class]&&isfinite([value doubleValue]);}

static void animate(CALayer *layer,NSString *key,NSArray *times,NSArray *values,double seconds) {
    CAKeyframeAnimation *a=[CAKeyframeAnimation animationWithKeyPath:key];
    a.keyTimes=times;a.values=values;a.calculationMode=kCAAnimationDiscrete;
    a.beginTime=AVCoreAnimationBeginTimeAtZero;a.duration=seconds;
    a.removedOnCompletion=NO;a.fillMode=kCAFillModeBoth;[layer addAnimation:a forKey:key];
}

static void marker(CALayer *parent,NSArray *rows,int kind) {
    // Rows contain top-left canvas points. Missing positions have nulls.
    int column=3+kind*2;BOOL final=kind>=2,pitch=(kind%2)==1;
    CAShapeLayer *layer=[CAShapeLayer layer];double radius=final?(pitch?7.:10.):(pitch?8.:8.);
    CGPathRef path=CGPathCreateWithEllipseInRect(CGRectMake(-radius,-radius,2*radius,2*radius),NULL);
    layer.path=path;CGPathRelease(path);layer.opacity=0.;layer.lineWidth=final?(pitch?1.5:2.8):2.;
    NSArray *palette=@[(__bridge id)NSColor.grayColor.CGColor,
        (__bridge id)[NSColor colorWithRed:.98 green:.98 blue:.98 alpha:1].CGColor,
        (__bridge id)[NSColor colorWithRed:1 green:.64 blue:.23 alpha:1].CGColor,
        (__bridge id)[NSColor colorWithRed:.25 green:.82 blue:.98 alpha:1].CGColor,
        (__bridge id)[NSColor colorWithRed:1 green:.86 blue:.32 alpha:1].CGColor];
    layer.fillColor=final&&pitch?NSColor.greenColor.CGColor:NSColor.clearColor.CGColor;
    layer.strokeColor=final&&pitch?NSColor.blackColor.CGColor:[NSColor colorWithWhite:.87 alpha:.95].CGColor;
    NSMutableArray *times=[NSMutableArray array],*positions=[NSMutableArray array],*opacity=[NSMutableArray array],*colors=[NSMutableArray array];
    double oldX=0,oldY=0;int oldColor=-1;BOOL oldVisible=NO;int count=(int)rows.count;
    for (int f=0;f<count;f++) {
        NSArray *row=rows[f];BOOL visible=number(row[column])&&number(row[column+1]);
        double x=visible?[row[column] doubleValue]:0.,y=visible?H-[row[column+1] doubleValue]:0.;
        int color=final?[row[1] intValue]:0;
        if (f && visible==oldVisible && (!visible || (x==oldX&&y==oldY&&color==oldColor)))continue;
        [times addObject:@(f?((double)f-.5)/count:0.)];
        [positions addObject:[NSValue valueWithPoint:NSMakePoint(x,y)]];[opacity addObject:@(visible?1.:0.)];
        [colors addObject:palette[MAX(0,MIN(4,color))]];
        oldX=x;oldY=y;oldVisible=visible;oldColor=color;
    }
    [times addObject:@1.];[positions addObject:positions.lastObject];[opacity addObject:opacity.lastObject];[colors addObject:colors.lastObject];
    animate(layer,@"position",times,positions,(double)count/FPS);animate(layer,@"opacity",times,opacity,(double)count/FPS);
    if (final)animate(layer,pitch?@"fillColor":@"strokeColor",times,colors,(double)count/FPS);
    [parent addSublayer:layer];
}

static void sprite(CALayer *parent,NSArray *rows,NSString *prefix,int column,CGRect rect) {
    CALayer *layer=[CALayer layer];layer.frame=CGRectMake(rect.origin.x,H-rect.origin.y-rect.size.height,rect.size.width,rect.size.height);
    NSMutableArray *times=[NSMutableArray array],*values=[NSMutableArray array];int count=(int)rows.count,previous=-99;
    for (int f=0;f<count;f++) {
        int code=[rows[f][column] intValue];if (f&&code==previous)continue;previous=code;
        [times addObject:@(f?((double)f-.5)/count:0.)];[values addObject:image([NSString stringWithFormat:@"%@_%d.png",prefix,code])];
    }
    [times addObject:@1.];[values addObject:values.lastObject];layer.contents=values[0];
    animate(layer,@"contents",times,values,(double)count/FPS);[parent addSublayer:layer];
}

static NSString *text(NSArray *row,int mode) {
    int frame=[row[0] intValue];
    if (mode==0)return [NSString stringWithFormat:@"%06d",frame];
    if (mode==1) {int sec=frame/FPS;return [NSString stringWithFormat:@"%02d:%02d:%02d.%02d",sec/3600,sec/60%60,sec%60,(frame%25)*4];}
    int column=mode==2?11:12;
    return number(row[column])?[NSString stringWithFormat:@"%+07.2f",[row[column] doubleValue]]:@" ---.--";
}

static void digits(CALayer *parent,NSArray *rows,int mode,int length,double x,double y) {
    int count=(int)rows.count;
    for (int digit=0;digit<length;digit++) {
        CALayer *layer=[CALayer layer];layer.frame=CGRectMake(x+digit*17,H-y-28,17,28);
        NSMutableArray *times=[NSMutableArray array],*values=[NSMutableArray array];unichar old=0;
        for (int f=0;f<count;f++) {
            NSString *value=text(rows[f],mode);unichar c=[value characterAtIndex:digit];if(f&&c==old)continue;old=c;
            [times addObject:@(f?((double)f-.5)/count:0.)];[values addObject:image([NSString stringWithFormat:@"glyph_%d.png",(int)c])];
        }
        [times addObject:@1.];[values addObject:values.lastObject];layer.contents=values[0];
        animate(layer,@"contents",times,values,(double)count/FPS);[parent addSublayer:layer];
    }
}


static void players(CALayer *parent,NSString *file,int count) {
    NSError *error=nil;
    __attribute__((objc_precise_lifetime)) NSData *data=[NSData dataWithContentsOfFile:file options:NSDataReadingMappedIfSafe error:&error];
    if(data.length!=(NSUInteger)count*46*3*sizeof(float)){NSLog(@"Player buffer %@",error);exit(11);}
    NSArray *styles=[NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:[assetDirectory stringByAppendingPathComponent:@"player_styles.json"]] options:0 error:&error];
    if(styles.count!=46){NSLog(@"Player styles %@",error);exit(12);}
    const float *points=(const float *)data.bytes;
    for(int slot=0;slot<46;slot++) {
        NSDictionary *style=styles[slot];
        CAShapeLayer *layer=[CAShapeLayer layer];
        BOOL keeper=[style[@"goalkeeper_slot"] boolValue];double radius=keeper?5.5:4.8;
        CGPathRef path=keeper?CGPathCreateWithRect(CGRectMake(-radius,-radius,2*radius,2*radius),NULL):CGPathCreateWithEllipseInRect(CGRectMake(-radius,-radius,2*radius,2*radius),NULL);
        layer.path=path;CGPathRelease(path);layer.lineWidth=1.1;layer.opacity=0.;
        BOOL home=[style[@"team_code"] isEqualToString:@"H"];
        layer.fillColor=(home?[NSColor colorWithRed:.22 green:.56 blue:.99 alpha:1]:[NSColor colorWithRed:1 green:.34 blue:.32 alpha:1]).CGColor;
        layer.strokeColor=[NSColor colorWithWhite:.94 alpha:1].CGColor;
        CALayer *label=[CALayer layer];label.frame=CGRectMake(5,3,34,18);
        label.contents=image([NSString stringWithFormat:@"slot_%02d.png",slot]);[layer addSublayer:label];
        NSMutableArray *times=[NSMutableArray array],*positions=[NSMutableArray array],*opacity=[NSMutableArray array];
        double oldX=0.,oldY=0.,oldOpacity=-1.;
        for(int f=0;f<count;f++) {
            NSUInteger k=((NSUInteger)f*46+slot)*3;
            BOOL visible=isfinite(points[k])&&isfinite(points[k+1]);
            double x=visible?points[k]:0.,y=visible?H-points[k+1]:0.;
            double alpha=visible?(points[k+2]>.5?1.:.62):0.;
            if(f&&alpha==oldOpacity&&(!visible||(x==oldX&&y==oldY)))continue;
            [times addObject:@(f?((double)f-.5)/count:0.)];
            [positions addObject:[NSValue valueWithPoint:NSMakePoint(x,y)]];[opacity addObject:@(alpha)];
            oldX=x;oldY=y;oldOpacity=alpha;
        }
        [times addObject:@1.];[positions addObject:positions.lastObject];[opacity addObject:opacity.lastObject];
        animate(layer,@"position",times,positions,(double)count/FPS);
        animate(layer,@"opacity",times,opacity,(double)count/FPS);
        [parent addSublayer:layer];
    }
}

int main(int argc,const char *argv[]) { @autoreleasepool {
    if(argc!=6){fprintf(stderr,"source rows_json background players_float32 output\n");return 2;}
    NSString *source=@(argv[1]),*metadata=@(argv[2]),*background=@(argv[3]),*playerFile=@(argv[4]),*target=@(argv[5]);
    assetDirectory=[background stringByDeletingLastPathComponent];imageCache=[NSMutableDictionary dictionary];NSError *error=nil;
    NSArray *rows=[NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:metadata] options:0 error:&error];
    if (!rows.count){NSLog(@"Metadata %@",error);return 3;}int count=(int)rows.count;
    if ([[NSFileManager defaultManager] fileExistsAtPath:target]){fprintf(stderr,"Refusing existing video\n");return 4;}
    AVURLAsset *asset=[AVURLAsset URLAssetWithURL:[NSURL fileURLWithPath:source] options:@{AVURLAssetPreferPreciseDurationAndTimingKey:@YES}];
    AVAssetTrack *track=[[asset tracksWithMediaType:AVMediaTypeVideo] firstObject];if(!track)return 5;
    AVAssetTrack *audio=[[asset tracksWithMediaType:AVMediaTypeAudio] firstObject];
    AVMutableComposition *composition=[AVMutableComposition composition];
    AVMutableCompositionTrack *video=[composition addMutableTrackWithMediaType:AVMediaTypeVideo preferredTrackID:kCMPersistentTrackID_Invalid];
    AVMutableCompositionTrack *sound=audio?[composition addMutableTrackWithMediaType:AVMediaTypeAudio preferredTrackID:kCMPersistentTrackID_Invalid]:nil;
    int pieces=0;
    for (int a=0;a<count;) {
        int b=a+1;while(b<count && [rows[b][0] intValue]==[rows[b-1][0] intValue]+1)b++;
        CMTimeRange range=CMTimeRangeMake(CMTimeMake([rows[a][0] intValue],FPS),CMTimeMake(b-a,FPS));CMTime at=CMTimeMake(a,FPS);
        if(![video insertTimeRange:range ofTrack:track atTime:at error:&error]){NSLog(@"Video insert %@",error);return 6;}
        if(sound && ![sound insertTimeRange:range ofTrack:audio atTime:at error:&error]){NSLog(@"Audio insert %@",error);return 7;}
        pieces++;a=b;
    }
    CMTime duration=CMTimeMake(count,FPS);
    AVMutableVideoComposition *vc=[AVMutableVideoComposition videoComposition];vc.renderSize=CGSizeMake(W,H);vc.frameDuration=CMTimeMake(1,FPS);
    AVMutableVideoCompositionInstruction *instruction=[AVMutableVideoCompositionInstruction videoCompositionInstruction];
    instruction.timeRange=CMTimeRangeMake(kCMTimeZero,duration);
    AVMutableVideoCompositionLayerInstruction *videoInstruction=[AVMutableVideoCompositionLayerInstruction videoCompositionLayerInstructionWithAssetTrack:video];
    [videoInstruction setTransform:CGAffineTransformMake(.5,0,0,.5,0,100) atTime:kCMTimeZero];
    instruction.layerInstructions=@[videoInstruction];vc.instructions=@[instruction];
    CALayer *parent=[CALayer layer],*videoLayer=[CALayer layer],*bg=[CALayer layer];
    parent.frame=CGRectMake(0,0,W,H);videoLayer.frame=parent.frame;bg.frame=parent.frame;bg.contents=image(background.lastPathComponent);
    [parent addSublayer:videoLayer];[parent addSublayer:bg];
    players(parent,playerFile,count);marker(parent,rows,2);marker(parent,rows,3);
    sprite(parent,rows,@"source",1,CGRectMake(18,655,930,34));
    sprite(parent,rows,@"state",2,CGRectMake(18,697,930,30));
    sprite(parent,rows,@"projection",13,CGRectMake(18,846,930,28));
    digits(parent,rows,0,6,1050,68);digits(parent,rows,1,11,1240,68);
    digits(parent,rows,2,7,1060,842);digits(parent,rows,3,7,1550,842);
    vc.animationTool=[AVVideoCompositionCoreAnimationTool videoCompositionCoreAnimationToolWithPostProcessingAsVideoLayer:videoLayer inLayer:parent];
    AVAssetExportSession *session=[[AVAssetExportSession alloc] initWithAsset:composition presetName:AVAssetExportPresetHighestQuality];
    session.outputURL=[NSURL fileURLWithPath:target];session.outputFileType=AVFileTypeMPEG4;session.videoComposition=vc;session.shouldOptimizeForNetworkUse=YES;
    dispatch_semaphore_t finish=dispatch_semaphore_create(0);NSDate *start=[NSDate date];
    [session exportAsynchronouslyWithCompletionHandler:^{dispatch_semaphore_signal(finish);}];
    while(dispatch_semaphore_wait(finish,dispatch_time(DISPATCH_TIME_NOW,5*NSEC_PER_SEC))!=0) {
        NSData *progress=[NSJSONSerialization dataWithJSONObject:@{@"status":@"rendering",@"progress":@(session.progress),@"frames":@(count),@"elapsed_seconds":@(-start.timeIntervalSinceNow)} options:NSJSONWritingPrettyPrinted error:nil];
        [progress writeToFile:[target stringByAppendingString:@".progress.json"] atomically:YES];
    }
    if(session.status!=AVAssetExportSessionStatusCompleted){NSLog(@"Export %@",session.error);return 8;}
    AVURLAsset *output=[AVURLAsset URLAssetWithURL:[NSURL fileURLWithPath:target] options:@{AVURLAssetPreferPreciseDurationAndTimingKey:@YES}];
    NSUInteger audioTracks=[output tracksWithMediaType:AVMediaTypeAudio].count;
    NSData *done=[NSJSONSerialization dataWithJSONObject:@{@"status":@"complete",@"frames":@(count),@"fps":@(FPS),@"elapsed_seconds":@(-start.timeIntervalSinceNow),@"audio_preserved":@(audio!=nil&&audioTracks>0),@"output_audio_tracks":@(audioTracks),@"output_duration_seconds":@(CMTimeGetSeconds(output.duration)),@"source_segments":@(pieces)} options:NSJSONWritingPrettyPrinted error:nil];
    [done writeToFile:[target stringByAppendingString:@".progress.json"] atomically:YES];return 0;
}}
