document.addEventListener('DOMContentLoaded', function () {
    const heroContainer = document.querySelector('.hero-container');
    if (!heroContainer) return;

    // Respect user's motion preferences for accessibility
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (prefersReducedMotion) {
        // Skip animation entirely for users who prefer reduced motion
        return;
    }

    // Create canvas
    const canvas = document.createElement('canvas');
    canvas.id = 'hero-canvas';
    heroContainer.insertBefore(canvas, heroContainer.firstChild);

    // Style canvas to sit behind content
    canvas.style.position = 'absolute';
    canvas.style.top = '0';
    canvas.style.left = '0';
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    canvas.style.opacity = '0.6'; // More visible against dark background
    canvas.style.pointerEvents = 'none'; // Let clicks pass through

    // Scene setup
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, heroContainer.clientWidth / heroContainer.clientHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ canvas: canvas, alpha: true, antialias: true });

    renderer.setSize(heroContainer.clientWidth, heroContainer.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);

    // Particles
    const particlesGeometry = new THREE.BufferGeometry();
    const particlesCount = 100;
    const posArray = new Float32Array(particlesCount * 3);

    for (let i = 0; i < particlesCount * 3; i++) {
        posArray[i] = (Math.random() - 0.5) * 20; // Spread heavily
    }

    particlesGeometry.setAttribute('position', new THREE.BufferAttribute(posArray, 3));

    const material = new THREE.PointsMaterial({
        size: 0.12,
        color: 0x38BDF8, // Sky blue particles matching theme accent
        transparent: true,
        opacity: 0.7
    });

    // Lines
    const linesMaterial = new THREE.LineBasicMaterial({
        color: 0x38BDF8,
        transparent: true,
        opacity: 0.2
    });

    const particlesMesh = new THREE.Points(particlesGeometry, material);
    scene.add(particlesMesh);

    camera.position.z = 5;

    // Animation variables
    let mouseX = 0;
    let mouseY = 0;

    // Mouse interaction
    document.addEventListener('mousemove', (event) => {
        mouseX = event.clientX / window.innerWidth - 0.5;
        mouseY = event.clientY / window.innerHeight - 0.5;
    });

    // Handle Resize
    window.addEventListener('resize', () => {
        camera.aspect = heroContainer.clientWidth / heroContainer.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(heroContainer.clientWidth, heroContainer.clientHeight);
    });

    // Animate
    function animate() {
        requestAnimationFrame(animate);

        // Gentle rotation
        particlesMesh.rotation.y += 0.001;
        particlesMesh.rotation.x += 0.001;

        // Mouse influence
        particlesMesh.rotation.y += mouseX * 0.01;
        particlesMesh.rotation.x += mouseY * 0.01;

        // Dynamic lines connecting close particles
        // (Simplified for performance: just rotating the cloud of points is often enough for a hero)
        // If we want actual lines, we need to rebuild geometry every frame or use a shader. 
        // For a hero background, a rotating point cloud is usually sufficient. 
        // Let's add a wave effect instead of lines for uniqueness.

        const positions = particlesGeometry.attributes.position.array;
        const time = Date.now() * 0.001;

        for (let i = 0; i < particlesCount; i++) {
            const i3 = i * 3;
            // Add subtle wave motion
            positions[i3 + 1] += Math.sin(time + positions[i3]) * 0.002;
        }
        particlesGeometry.attributes.position.needsUpdate = true;

        renderer.render(scene, camera);
    }

    animate();
});
