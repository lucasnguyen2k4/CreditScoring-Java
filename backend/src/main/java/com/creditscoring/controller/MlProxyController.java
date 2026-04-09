package com.creditscoring.controller;

import jakarta.servlet.http.HttpServletRequest;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.http.client.MultipartBodyBuilder;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.client.WebClientResponseException;

import java.util.UUID;

/**
 * Proxies all /api/ml/** requests to the Python ML service.
 * Adds X-Session-ID header for session isolation.
 * Uses blocking calls (.block()) for proper Spring Security integration.
 */
@RestController
@RequestMapping("/api/ml")
@RequiredArgsConstructor
public class MlProxyController {

    private final WebClient mlServiceWebClient;

    /**
     * Get session ID — use authenticated username as session ID
     * so each user has their own ML session.
     */
    private String getSessionId() {
        var auth = SecurityContextHolder.getContext().getAuthentication();
        return auth != null ? auth.getName() : UUID.randomUUID().toString();
    }

    // ==================== FILE UPLOAD (special handling for multipart) ====================

    @PostMapping("/data/upload")
    public ResponseEntity<String> uploadFile(@RequestParam("file") MultipartFile file) {
        MultipartBodyBuilder builder = new MultipartBodyBuilder();
        builder.part("file", file.getResource())
               .filename(file.getOriginalFilename() != null ? file.getOriginalFilename() : "upload.csv");

        try {
            ResponseEntity<String> response = mlServiceWebClient.post()
                    .uri("/api/ml/data/upload")
                    .header("X-Session-ID", getSessionId())
                    .contentType(MediaType.MULTIPART_FORM_DATA)
                    .body(BodyInserters.fromMultipartData(builder.build()))
                    .retrieve()
                    .toEntity(String.class)
                    .block(java.time.Duration.ofMinutes(5));
            return response;
        } catch (WebClientResponseException e) {
            return ResponseEntity.status(e.getStatusCode()).body(e.getResponseBodyAsString());
        } catch (Exception e) {
            return ResponseEntity.internalServerError().body("{\"error\":\"" + e.getMessage() + "\"}");
        }
    }

    // ==================== GENERIC GET PROXY ====================

    @GetMapping("/data/**")
    public ResponseEntity<String> proxyDataGet(HttpServletRequest request) {
        return proxyGet(request);
    }

    @GetMapping("/model/**")
    public ResponseEntity<String> proxyModelGet(HttpServletRequest request) {
        return proxyGet(request);
    }

    @GetMapping("/predict/**")
    public ResponseEntity<String> proxyPredictGet(HttpServletRequest request) {
        return proxyGet(request);
    }

    @GetMapping("/shap/**")
    public ResponseEntity<String> proxyShapGet(HttpServletRequest request) {
        return proxyGet(request);
    }

    @GetMapping("/llm/**")
    public ResponseEntity<String> proxyLlmGet(HttpServletRequest request) {
        return proxyGet(request);
    }

    // ==================== GENERIC POST PROXY ====================

    @PostMapping("/data/{action}")
    public ResponseEntity<String> proxyDataPost(
            @PathVariable String action,
            @RequestBody(required = false) String body,
            HttpServletRequest request) {
        if ("upload".equals(action)) {
            // Upload is handled separately above
            return ResponseEntity.badRequest().body("{\"error\":\"Use multipart upload endpoint\"}");
        }
        return proxyPost(request, body);
    }

    @PostMapping("/model/**")
    public ResponseEntity<String> proxyModelPost(
            @RequestBody(required = false) String body, HttpServletRequest request) {
        return proxyPost(request, body);
    }

    @PostMapping("/predict/**")
    public ResponseEntity<String> proxyPredictPost(
            @RequestBody(required = false) String body, HttpServletRequest request) {
        return proxyPost(request, body);
    }

    @PostMapping("/shap/**")
    public ResponseEntity<String> proxyShapPost(
            @RequestBody(required = false) String body, HttpServletRequest request) {
        return proxyPost(request, body);
    }

    @PostMapping("/llm/**")
    public ResponseEntity<String> proxyLlmPost(
            @RequestBody(required = false) String body, HttpServletRequest request) {
        return proxyPost(request, body);
    }

    // ==================== PROXY HELPERS ====================

    private ResponseEntity<String> proxyGet(HttpServletRequest request) {
        String path = request.getRequestURI();  // e.g., /api/ml/data/info
        String query = request.getQueryString();
        String fullUri = query != null ? path + "?" + query : path;

        try {
            ResponseEntity<String> response = mlServiceWebClient.get()
                    .uri(fullUri)
                    .header("X-Session-ID", getSessionId())
                    .retrieve()
                    .toEntity(String.class)
                    .block();
            return response;
        } catch (WebClientResponseException e) {
            return ResponseEntity.status(e.getStatusCode()).body(e.getResponseBodyAsString());
        } catch (Exception e) {
            return ResponseEntity.internalServerError().body("{\"error\":\"" + e.getMessage() + "\"}");
        }
    }

    private ResponseEntity<String> proxyPost(HttpServletRequest request, String body) {
        String path = request.getRequestURI();
        String query = request.getQueryString();
        String fullUri = query != null ? path + "?" + query : path;

        try {
            var req = mlServiceWebClient.post()
                    .uri(fullUri)
                    .header("X-Session-ID", getSessionId())
                    .contentType(MediaType.APPLICATION_JSON);

            ResponseEntity<String> response;
            if (body != null && !body.isEmpty()) {
                response = req.bodyValue(body).retrieve().toEntity(String.class).block();
            } else {
                response = req.retrieve().toEntity(String.class).block();
            }
            return response;
        } catch (WebClientResponseException e) {
            return ResponseEntity.status(e.getStatusCode()).body(e.getResponseBodyAsString());
        } catch (Exception e) {
            return ResponseEntity.internalServerError().body("{\"error\":\"" + e.getMessage() + "\"}");
        }
    }
}
