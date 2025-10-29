# Changelog - What's New?

All the cool stuff we've added to the project! 🎉

---

## Version 0.2.0 - October 24, 2025 🚀

### 🎉 Big News: Phase 2 Complete!

We finished Phase 2 way ahead of schedule (23 days early)! The search system is now fully working!

### ✨ What's New

**The Search Engine Works!**
- ✅ Search for images using text ("show me dogs")
- ✅ Get captions for any image
- ✅ Find similar-looking images
- ✅ Process multiple searches at once (batch mode)
- ✅ Super fast: 11 milliseconds per search!

**New Files You Can Use:**
- `src/retrieval/bi_encoder.py` - The AI that understands images and text
- `src/retrieval/faiss_index.py` - The fast search engine
- `src/retrieval/search_engine.py` - The easy-to-use interface
- `notebooks/05_search_demo.ipynb` - Try it yourself! (Interactive)

**Generated Data:**
- Embeddings for all 31,783 images (saved in `data/embeddings/`)
- Embeddings for all 158,914 captions
- Fast search indices (saved in `data/indices/`)

**Helper Scripts:**
- `scripts/build_faiss_indices.py` - Build the search database
- `scripts/test_search_engine.py` - Make sure everything works

### 📊 How Fast Is It?

- **Search:** 11ms (that's 0.011 seconds!)
- **First search:** 443ms (loads the AI model first)
- **Can search:** 31,783 images almost instantly

**Target was 100ms. We got 11ms. That's 9x faster than needed!** 🎯

### 🤖 What's Under the Hood

- Uses CLIP AI model (ViT-B/32)
- FAISS for super-fast search
- Works on GPU (CUDA) for speed
- Python 3.13.7
- PyTorch 2.9.0

**Don't worry if you don't understand this - it just works!** ✨

---

## Version 0.1.0 - October 20, 2025 📦

### 🎉 First Release: Project Started!

**What We Set Up:**
- ✅ Downloaded 31,783 images from Flickr30K
- ✅ Set up Python environment
- ✅ Created project structure
- ✅ Made tools to load and explore the data

**New Files:**
- `src/flickr30k/dataset.py` - Load images and captions
- `notebooks/flickr30k_exploration.ipynb` - Explore the data
- `scripts/download_flickr30k.py` - Download the dataset
- `README.md` - How to use this project
- `IMPLEMENTATION_PLAN.md` - What we're going to build

**The Data:**
- 31,783 images
- 158,915 captions (about 5 for each image)
- From Flickr (people's vacation photos!)

---

## 🔮 What's Coming Next?

### Phase 3 (November 2025) - Making Search Smarter
- Add BLIP-2 AI model
- Make search results even more accurate
- Hybrid search (combines two AI models)

### Phase 4 (December 2025) - Knowledge Graphs
- Connect related images and captions
- Understand context better
- Smarter recommendations

### Phase 5 (January-February 2026) - Final Polish
- Add AI that explains results in natural language
- Complete evaluation
- Write the final thesis
- Present the project

---

## 📈 Project Status

| Phase | Status | When |
|-------|--------|------|
| Phase 1: Setup | ✅ Done | Oct 2025 |
| Phase 2: Search Engine | ✅ Done | Oct 2025 |
| Phase 3: Smarter Search | 🚧 Next | Nov 2025 |
| Phase 4: Knowledge Graph | 📋 Planned | Dec 2025 |
| Phase 5: Final | 📋 Planned | Jan-Feb 2026 |

**Current Progress:** 40% complete (2 out of 5 phases done!)

---

## 🎯 Legend

What do these symbols mean?

- ✨ New feature - Something cool we added
- 🔧 Changed - We improved something that existed
- 🐛 Fixed - We fixed a bug
- 📚 Documentation - Better guides and docs
- 📈 Performance - Made it faster
- 🎉 Milestone - Big achievement!

---

**Last Updated:** October 24, 2025  
**Version:** 0.2.0  
**Next Update:** When Phase 3 is done!

---

**Want to see what we're working on?** Check [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)
