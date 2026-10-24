#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#import <UIKit/UIKit.h>

#include <chrono>
#include <cmath>
#include "generated/turing_program_abi.h"

@interface TuringView : MTKView
@property(nonatomic) float touchX;
@property(nonatomic) float touchY;
@end

@implementation TuringView
- (instancetype)initWithFrame:(CGRect)frame device:(id<MTLDevice>)device {
    self = [super initWithFrame:frame device:device];
    if (self) { _touchX = 0.5f; _touchY = 0.5f; self.multipleTouchEnabled = YES; }
    return self;
}
- (void)touchesMoved:(NSSet<UITouch*>*)touches withEvent:(UIEvent*)event {
    CGPoint point = [[touches anyObject] locationInView:self];
    _touchX = fmaxf(0, fminf(1, point.x / self.bounds.size.width));
    _touchY = fmaxf(0, fminf(1, point.y / self.bounds.size.height));
}
@end

@interface Renderer : NSObject<MTKViewDelegate>
- (instancetype)initWithView:(TuringView*)view;
@end

@implementation Renderer {
    TuringView* _view;
    id<MTLCommandQueue> _queue;
    id<MTLRenderPipelineState> _pipeline;
    std::chrono::steady_clock::time_point _start;
}
- (instancetype)initWithView:(TuringView*)view {
    self = [super init];
    if (!self) return nil;
    _view = view;
    _queue = [view.device newCommandQueue];
    NSError* error = nil;
    NSString* vertexPath = [[NSBundle mainBundle] pathForResource:@"fullscreen.vert" ofType:@"metal"];
    NSString* fragmentPath = [[NSBundle mainBundle] pathForResource:@"main_output.frag" ofType:@"metal"];
    NSString* vertexSource = [NSString stringWithContentsOfFile:vertexPath encoding:NSUTF8StringEncoding error:&error];
    id<MTLLibrary> vertexLibrary = [view.device newLibraryWithSource:vertexSource options:nil error:&error];
    NSAssert(vertexLibrary, @"vertex Metal compile failed: %@", error);
    NSString* fragmentSource = [NSString stringWithContentsOfFile:fragmentPath encoding:NSUTF8StringEncoding error:&error];
    id<MTLLibrary> fragmentLibrary = [view.device newLibraryWithSource:fragmentSource options:nil error:&error];
    NSAssert(fragmentLibrary, @"fragment Metal compile failed: %@", error);
    MTLRenderPipelineDescriptor* descriptor = [MTLRenderPipelineDescriptor new];
    descriptor.vertexFunction = [vertexLibrary newFunctionWithName:@"main0"];
    descriptor.fragmentFunction = [fragmentLibrary newFunctionWithName:@"main0"];
    descriptor.colorAttachments[0].pixelFormat = view.colorPixelFormat;
    _pipeline = [view.device newRenderPipelineStateWithDescriptor:descriptor error:&error];
    NSAssert(_pipeline, @"pipeline creation failed: %@", error);
    _start = std::chrono::steady_clock::now();
    return self;
}
- (void)mtkView:(MTKView*)view drawableSizeWillChange:(CGSize)size {}
- (void)drawInMTKView:(MTKView*)view {
    if (!view.currentDrawable || !view.currentRenderPassDescriptor) return;
    float elapsed = std::chrono::duration<float>(std::chrono::steady_clock::now() - _start).count();
    double phase = 0.5 + 0.5 * sin(elapsed * 0.7);
    TuringMobileColor result = turing_mobile_eval(phase, _view.touchX, _view.touchY);
    simd_float4 color = {(float)result.red, (float)result.green, (float)result.blue, 1};
    id<MTLCommandBuffer> commands = [_queue commandBuffer];
    id<MTLRenderCommandEncoder> encoder = [commands renderCommandEncoderWithDescriptor:view.currentRenderPassDescriptor];
    [encoder setRenderPipelineState:_pipeline];
    [encoder setFragmentBytes:&color length:sizeof(color) atIndex:0];
    [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [encoder endEncoding];
    [commands presentDrawable:view.currentDrawable];
    [commands commit];
}
@end

@interface AppDelegate : UIResponder<UIApplicationDelegate>
@property(nonatomic, strong) UIWindow* window;
@property(nonatomic, strong) Renderer* renderer;
@end

@implementation AppDelegate
- (BOOL)application:(UIApplication*)application didFinishLaunchingWithOptions:(NSDictionary*)options {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSAssert(device, @"Metal is required");
    self.window = [[UIWindow alloc] initWithFrame:UIScreen.mainScreen.bounds];
    UIViewController* controller = [UIViewController new];
    TuringView* view = [[TuringView alloc] initWithFrame:self.window.bounds device:device];
    view.autoresizingMask = UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight;
    view.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
    view.preferredFramesPerSecond = 60;
    self.renderer = [[Renderer alloc] initWithView:view];
    view.delegate = self.renderer;
    controller.view = view;
    self.window.rootViewController = controller;
    [self.window makeKeyAndVisible];
    return YES;
}
@end

int main(int argc, char** argv) {
    @autoreleasepool { return UIApplicationMain(argc, argv, nil, NSStringFromClass(AppDelegate.class)); }
}
