import { useEffect, useRef, useCallback, useState } from 'react';
import { FilesetResolver, GestureRecognizer } from '@mediapipe/tasks-vision';

export const useContactless = (callback, { runningMode = 'VIDEO', numHands = 2, swipeThreshold = 0.10 } = {}) => {
  const recognizerRef = useRef(null);
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const isRunning = useRef(false);
  const [isReady, setIsReady] = useState(false);

  // Advanced gesture tracker state
  const tracking = useRef({
    historyX: [],
    historyY: [],
    swipeAccumulator: 0,
    lastGestureName: '',
    lastOpenPalmTime: 0,
    cooldownEnd: 0,
    smoothedX: null
  });

  const initHandTracking = useCallback(async () => {
    try {
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
      );
      const recognizer = await GestureRecognizer.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
          delegate: "GPU"
        },
        runningMode: runningMode,
        numHands: numHands
      });
      recognizerRef.current = recognizer;
      setIsReady(true);
    } catch (e) {
      console.error("Failed to initialize MediaPipe", e);
    }
  }, [numHands, runningMode]);

  const stopCamera = useCallback(() => {
    isRunning.current = false;
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }, []);

  const processComplexGestures = useCallback((results) => {
    if (!results.landmarks || results.landmarks.length === 0) {
       tracking.current.historyX = [];
       tracking.current.historyY = [];
       tracking.current.swipeAccumulator = 0;
       tracking.current.lastGestureName = '';
       tracking.current.smoothedX = null;
       return null;
    }

    const now = Date.now();
    let semanticGesture = null;

    if (now > tracking.current.cooldownEnd) {
      
      // 1. Check for two-handed PRANAM pose (Namaste)
      if (results.landmarks.length === 2) {
          const hand1 = results.landmarks[0];
          const hand2 = results.landmarks[1];
          const indexDist = Math.hypot(hand1[8].x - hand2[8].x, hand1[8].y - hand2[8].y);
          const wristDist = Math.hypot(hand1[0].x - hand2[0].x, hand1[0].y - hand2[0].y);
          
          if (indexDist < 0.1 && wristDist < 0.15) {
              semanticGesture = 'PRANAM';
              tracking.current.cooldownEnd = now + 1500;
              return semanticGesture; 
          }
      }

      // 2. Single hand Gestures mapping
      const gestureObj = results.gestures && results.gestures.length > 0 ? results.gestures[0][0] : null;
      const currentName = gestureObj ? gestureObj.categoryName : 'None';

      if (currentName !== tracking.current.lastGestureName) {
        switch (currentName) {
          case 'Closed_Fist':
            semanticGesture = 'FIST';
            tracking.current.cooldownEnd = now + 500;
            break;
          case 'Pointing_Up':
            semanticGesture = 'ONE_FINGER';
            tracking.current.cooldownEnd = now + 500;
            break;
          case 'Victory':
            semanticGesture = 'TWO_FINGERS';
            tracking.current.cooldownEnd = now + 500;
            break;
          case 'Thumb_Up':
            semanticGesture = 'THUMBS_UP';
            tracking.current.cooldownEnd = now + 500;
            break;
          default:
            break;
        }
      }
      
      // 3. Simple & Robust Swipe Detection (4 directions) - ONLY with Open_Palm
      const rawX = results.landmarks[0][0].x; 
      const rawY = results.landmarks[0][0].y; 
      
      tracking.current.historyX.push({ val: rawX, time: now });
      tracking.current.historyY.push({ val: rawY, time: now });
      if (tracking.current.historyX.length > 15) tracking.current.historyX.shift();
      if (tracking.current.historyY.length > 15) tracking.current.historyY.shift();

      if (tracking.current.historyX.length >= 3 && !semanticGesture && currentName === 'Open_Palm') {
         const oldX = tracking.current.historyX[0];
         const oldY = tracking.current.historyY[0];
         const deltaX = rawX - oldX.val;
         const deltaY = rawY - oldY.val;
         const deltaTime = now - oldX.time;
         
         // If a quick movement occurs (under 400ms)
         if (deltaTime > 0 && deltaTime < 400) { 
             const absX = Math.abs(deltaX);
             const absY = Math.abs(deltaY);
             
             // Check if movement is significant enough
             if (absX > swipeThreshold || absY > swipeThreshold) {
                // Determine dominant direction
                if (absX > absY) {
                   // Horizontal swipe
                   semanticGesture = deltaX > 0 ? 'SWIPE_RIGHT' : 'SWIPE_LEFT'; 
                } else {
                   // Vertical swipe
                   semanticGesture = deltaY > 0 ? 'SWIPE_DOWN' : 'SWIPE_UP'; 
                }
                
                tracking.current.historyX = [];
                tracking.current.historyY = [];
                tracking.current.cooldownEnd = now + 700;
             }
         }
      }

      tracking.current.lastGestureName = currentName;
    }

    return semanticGesture;
  }, [swipeThreshold]);

  const predictWebcam = useCallback(() => {
    if (!recognizerRef.current || !videoRef.current) return;
    
    isRunning.current = true;
    let lastVideoTime = -1;

    const tick = () => {
      if (!isRunning.current) return;
        
      if (videoRef.current.currentTime !== lastVideoTime && videoRef.current.videoWidth > 0 && videoRef.current.videoHeight > 0) {
        lastVideoTime = videoRef.current.currentTime;
        try {
          const results = recognizerRef.current.recognizeForVideo(videoRef.current, performance.now());
          const semanticGesture = processComplexGestures(results);
          
          if (callback) {
             callback({ ...results, semanticGesture });
          }
        } catch (err) {
          console.error("Inference Error:", err);
        }
      }
      requestAnimationFrame(tick);
    }
    tick();
  }, [callback, processComplexGestures]);

  const startCamera = useCallback(async (videoElement) => {
    videoRef.current = videoElement;
    if (!videoElement) return;
    
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = stream;
      streamRef.current = stream;
      videoRef.current.onloadedmetadata = () => {
        videoRef.current.play();
        predictWebcam();
      };
    } catch (e) {
      console.error("Error accessing camera", e);
    }
  }, [predictWebcam]);

  useEffect(() => {
    const init = async () => {
      await initHandTracking();
    };
    init();
    return () => stopCamera(); 
  }, [initHandTracking, stopCamera]);

  return { start: startCamera, stop: stopCamera, isReady };
};
